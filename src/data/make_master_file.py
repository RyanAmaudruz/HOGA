### script for writing meta information of datasets into master.csv
### for node property prediction datasets.
import pandas as pd
import os.path as osp
import os
import argparse

_DATA_DIR = osp.dirname(osp.abspath(__file__))


def make_master(design_name, num_class=5, new=0):
    master_path = osp.join(_DATA_DIR, "master.csv")
    if not new and osp.exists(master_path):
        dataset_dict = pd.read_csv(master_path, index_col=0).to_dict()
        name = design_name
        dataset_dict[name] = {
            "num tasks": 1,
            "num classes": num_class,
            "eval metric": "acc",
            "task type": "multiclass classification",
        }
        dataset_dict[name]["download_name"] = design_name
        dataset_dict[name]["version"] = 1
        dataset_dict[name]["url"] = None
        dataset_dict[name]["add_inverse_edge"] = False
        dataset_dict[name]["has_node_attr"] = True
        dataset_dict[name]["has_edge_attr"] = False
        dataset_dict[name]["split"] = "Random"
        dataset_dict[name]["additional node files"] = "None"
        dataset_dict[name]["additional edge files"] = "None"
        dataset_dict[name]["is hetero"] = False
        dataset_dict[name]["binary"] = False
    else:
        dataset_dict = {}
        name = design_name
        dataset_dict[name] = {
            "num tasks": 1,
            "num classes": num_class,
            "eval metric": "acc",
            "task type": "multiclass classification",
        }
        dataset_dict[name]["download_name"] = design_name
        dataset_dict[name]["version"] = 1
        dataset_dict[name]["url"] = None
        dataset_dict[name]["add_inverse_edge"] = False
        dataset_dict[name]["has_node_attr"] = True
        dataset_dict[name]["has_edge_attr"] = False
        dataset_dict[name]["split"] = "Random"
        dataset_dict[name]["additional node files"] = "None"
        dataset_dict[name]["additional edge files"] = "None"
        dataset_dict[name]["is hetero"] = False
        dataset_dict[name]["binary"] = False

    df = pd.DataFrame(dataset_dict)
    df.to_csv(master_path)
    return


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--design_name", type=str, default="")
    parser.add_argument("--new", type=int, default=0)
    parser.add_argument("--num_class", type=int, default=5)
    args = parser.parse_args()
    make_master(args.design_name, args.num_class, new=args.new)


if __name__ == "__main__":
    main()
