import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import torch_geometric.transforms as T

from src.models.hoga import HOGA
from src.data import PygNodePropPredDataset, Evaluator
from src.training import train, test_all
from src.utils import Logger, preprocess


torch.manual_seed(0)


def main():
    parser = argparse.ArgumentParser(description="mult16")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--bits_test", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_hops", type=int, default=8)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--mapped", type=int, default=1)
    parser.add_argument("--lda1", type=int, default=5)
    parser.add_argument("--lda2", type=int, default=1)
    parser.add_argument("--design", type=str, default="booth")
    parser.add_argument("--root_dir", type=str, default="/scratch-x3/circuit_datasets")
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--test_all_bits", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()
    print(args)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    if not os.path.exists("models/"):
        os.makedirs("models/")

    if args.mapped == 1:
        suffix = "_7nm_mapped"
    elif args.mapped == 2:
        suffix = "_mapped"
    else:
        suffix = ""

    if args.design == "booth":
        design_name = "booth_mult" + str(args.bits) + suffix
        root_path = f"{args.root_dir}/booth/"
    else:
        design_name = "mult" + str(args.bits) + suffix
        root_path = f"{args.root_dir}/csa/"
    train_design_name = design_name
    design_name_root = design_name + "_root"
    design_name_shared = design_name + "_shared"

    master_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "src", "data", "master.csv"
    )
    master = pd.read_csv(master_path, index_col=0)
    if design_name_root not in master:
        os.system(
            f"python -m src.data.make_master_file --design_name {design_name_root}"
        )
    if design_name_shared not in master:
        os.system(
            f"python -m src.data.make_master_file --design_name {design_name_shared}"
        )
    dataset_r = PygNodePropPredDataset(name=f"{design_name_root}", root=root_path)
    print("Training on %s" % design_name)
    data_r = dataset_r[0]
    data_r = T.ToSparseTensor()(data_r)

    dataset = PygNodePropPredDataset(name=f"{design_name_shared}", root=root_path)
    data = dataset[0]
    data = preprocess(data, args)
    data = T.ToSparseTensor()(data)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    valid_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    batch_data_train = Data.TensorDataset(
        data.x[train_idx], data.y[train_idx], data_r.y[train_idx]
    )
    batch_data_test = Data.TensorDataset(
        data.x[test_idx], data.y[test_idx], data_r.y[test_idx]
    )

    train_loader = Data.DataLoader(
        batch_data_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
    )
    test_loader = Data.DataLoader(
        batch_data_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=10,
    )

    model = HOGA(
        data.num_features,
        args.hidden_channels,
        3,
        args.num_layers,
        args.dropout,
        num_hops=args.num_hops + 1,
        heads=args.heads,
        attn_type="mix",
    ).to(device)

    logger_r = Logger(args.runs, args)
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        best_test_r = float("-inf")
        best_test_s = float("-inf")
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, train_loader, optimizer, device, args)
            result = test_all(model, test_loader, device)
            logger_r.add_result(run, result[:3])
            logger.add_result(run, result[3:])

            if epoch % args.log_steps == 0:
                train_acc_r, valid_acc_r, test_acc_r, train_acc_s, valid_acc_s, test_acc_s = (
                    result
                )
                if test_acc_s >= best_test_s:
                    best_test_r = test_acc_r
                    best_test_s = test_acc_s
                    if args.save_model:
                        model_name = f"models/hoga_{design_name}_{args.design}.pt"
                        torch.save({"model_state_dict": model.state_dict()}, model_name)
                print(
                    f"Run: {run + 1:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {loss:.4f}, "
                    f"[Root Model] Train: {100 * train_acc_r:.2f}%, "
                    f"[Root Model] Valid: {100 * valid_acc_r:.2f}% "
                    f"[Root Model] Test: {100 * test_acc_r:.2f}% "
                    f"[Shared Model] Train: {100 * train_acc_s:.2f}%, "
                    f"[Shared Model] Valid: {100 * valid_acc_s:.2f}% "
                    f"[Shared Model] Test: {100 * test_acc_s:.2f}%"
                )

        logger_r.print_statistics(run)
        logger.print_statistics(run)
    logger_r.print_statistics()
    logger.print_statistics()

    logger_eval_r = Logger(1, args)
    logger_eval = Logger(1, args)

    if args.mapped == 1:
        suffix = "_7nm_mapped"
    elif args.mapped == 2:
        suffix = "_mapped"
    else:
        suffix = ""

    if args.test_all_bits:
        bits_test_lst = [
            64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768
        ]
    else:
        bits_test_lst = [args.bits_test]

    if args.save_model:
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint["model_state_dict"])

    for bits_test in bits_test_lst:
        if args.design == "booth":
            design_name = "booth_mult" + str(bits_test) + suffix
        else:
            design_name = "mult" + str(bits_test) + suffix
        design_name_root = design_name + "_root"
        design_name_shared = design_name + "_shared"
        print("Evaluation on %s" % design_name)

        master = pd.read_csv(master_path, index_col=0)
        if design_name_root not in master:
            os.system(
                f"python -m src.data.make_master_file --design_name {design_name_root}"
            )
        if design_name_shared not in master:
            os.system(
                f"python -m src.data.make_master_file --design_name {design_name_shared}"
            )
        dataset_r = PygNodePropPredDataset(
            name=f"{design_name_root}", root=root_path
        )
        data_r = dataset_r[0]
        data_r = T.ToSparseTensor()(data_r)

        dataset = PygNodePropPredDataset(
            name=f"{design_name_shared}", root=root_path
        )
        data = dataset[0]
        data = preprocess(data, args)
        data = T.ToSparseTensor()(data)

        batch_data_test = Data.TensorDataset(data.x, data.y, data_r.y)
        test_loader = Data.DataLoader(
            batch_data_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=10,
        )

        for run_1 in range(1):
            for epoch in range(1):
                file_name = f"{args.design}_{design_name_shared}"
                result = test_all(model, test_loader, device, file_name)
                logger_eval_r.add_result(run_1, result[:3])
                logger_eval.add_result(run_1, result[3:])
                if epoch % args.log_steps == 0:
                    (
                        train_acc_r,
                        valid_acc_r,
                        test_acc_r,
                        train_acc_s,
                        valid_acc_s,
                        test_acc_s,
                    ) = result
                    print(
                        f"Run: {run_1 + 1:02d}, "
                        f"Epoch: {epoch:02d}, "
                        f"Loss: {loss:.4f}, "
                        f"[Root Model] Train: {100 * train_acc_r:.2f}%, "
                        f"[Root Model] Valid: {100 * valid_acc_r:.2f}% "
                        f"[Root Model] Test: {100 * test_acc_r:.2f}% "
                        f"[Shared Model] Train: {100 * train_acc_s:.2f}%, "
                        f"[Shared Model] Valid: {100 * valid_acc_s:.2f}% "
                        f"[Shared Model] Test: {100 * test_acc_s:.2f}%"
                    )

        logger_eval_r.print_statistics()
        logger_eval.print_statistics()

        if not os.path.exists("results/hoga"):
            os.makedirs("results/hoga")
        filename = f"results/hoga/{args.design}_{train_design_name}.csv"
        print(f"Saving results to {filename}")
        with open(filename, "a+") as write_obj:
            write_obj.write(
                f"{design_name} "
                + f"{args.weight_decay} "
                + f"{args.dropout} "
                + f"{args.lr} "
                + f"{args.num_layers} "
                + f"{args.epochs} "
                + f"{args.hidden_channels} "
                + f"test_acc_r: {100 * test_acc_r:.2f} "
                + f"test_acc_s: {100 * test_acc_s:.2f} \n"
            )


if __name__ == "__main__":
    main()
