from torch_geometric.data import InMemoryDataset
import pandas as pd
import os
import os.path as osp
import torch
import numpy as np

from src.data.io.read_graph_pyg import read_graph_pyg, read_heterograph_pyg
from src.data.io.read_graph_raw import (
    read_node_label_hetero,
    read_nodesplitidx_split_hetero,
)
from src.data.make_master_file import make_master


class PygNodePropPredDataset(InMemoryDataset):
    def __init__(
        self,
        name,
        num_class=0,
        root=None,
        transform=None,
        pre_transform=None,
        meta_dict=None,
    ):
        if root is None:
            root = osp.join(osp.dirname(osp.abspath(__file__)), "dataset")
        self.name = name
        if meta_dict is None:
            self.dir_name = "_".join(name.split("-"))
            if osp.exists(osp.join(root, self.dir_name + "_pyg")):
                self.dir_name = self.dir_name + "_pyg"

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            master = pd.read_csv(
                osp.join(osp.dirname(__file__), "master.csv"), index_col=0
            )
            if self.name not in master:
                error_mssg = "Invalid dataset name {}.\n".format(self.name)
                error_mssg += "Available datasets are as follows:\n"
                error_mssg += "\n".join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]
        else:
            self.dir_name = meta_dict["dir_path"]
            self.original_root = ""
            self.root = meta_dict["dir_path"]
            self.meta_info = meta_dict

        self.download_name = self.meta_info["download_name"]
        self.num_tasks = int(self.meta_info["num tasks"])
        self.task_type = self.meta_info["task type"]
        self.eval_metric = self.meta_info["eval metric"]
        self.__num_classes__ = int(self.meta_info["num classes"])
        self.is_hetero = self.meta_info["is hetero"] == "True"
        self.binary = self.meta_info["binary"] == "True"

        super(PygNodePropPredDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info["split"]

        path = osp.join(self.root, "split", split_type)

        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            return torch.load(os.path.join(path, "split_dict.pt"))

        if self.is_hetero:
            train_idx_dict, valid_idx_dict, test_idx_dict = read_nodesplitidx_split_hetero(
                path
            )
            for nodetype in train_idx_dict.keys():
                train_idx_dict[nodetype] = torch.from_numpy(
                    train_idx_dict[nodetype]
                ).to(torch.long)
                valid_idx_dict[nodetype] = torch.from_numpy(
                    valid_idx_dict[nodetype]
                ).to(torch.long)
                test_idx_dict[nodetype] = torch.from_numpy(
                    test_idx_dict[nodetype]
                ).to(torch.long)
                return {
                    "train": train_idx_dict,
                    "valid": valid_idx_dict,
                    "test": test_idx_dict,
                }
        else:
            train_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, "train.csv.gz"), compression="gzip", header=None).values.T[0]
            ).to(torch.long)
            valid_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, "valid.csv.gz"), compression="gzip", header=None).values.T[0]
            ).to(torch.long)
            test_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, "test.csv.gz"), compression="gzip", header=None).values.T[0]
            ).to(torch.long)
            return {"train": train_idx, "valid": valid_idx, "test": test_idx}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            if self.is_hetero:
                return ["edge_index_dict.npz"]
            else:
                return ["data.npz"]
        else:
            if self.is_hetero:
                return ["num-node-dict.csv.gz", "triplet-type-list.csv.gz"]
            else:
                file_names = ["edge"]
                if self.meta_info["has_node_attr"] == "True":
                    file_names.append("node-feat")
                if self.meta_info["has_edge_attr"] == "True":
                    file_names.append("edge-feat")
                return [file_name + ".csv.gz" for file_name in file_names]

    @property
    def processed_file_names(self):
        return osp.join("geometric_data_processed.pt")

    def process(self):
        add_inverse_edge = self.meta_info["add_inverse_edge"] == "True"

        if self.meta_info["additional node files"] == "None":
            additional_node_files = []
        else:
            additional_node_files = self.meta_info["additional node files"].split(",")

        if self.meta_info["additional edge files"] == "None":
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info["additional edge files"].split(",")

        if self.is_hetero:
            data = read_heterograph_pyg(
                self.raw_dir,
                add_inverse_edge=add_inverse_edge,
                additional_node_files=additional_node_files,
                additional_edge_files=additional_edge_files,
                binary=self.binary,
            )[0]

            if self.binary:
                tmp = np.load(osp.join(self.raw_dir, "node-label.npz"))
                node_label_dict = {key: tmp[key] for key in list(tmp.keys())}
                del tmp
            else:
                node_label_dict = read_node_label_hetero(self.raw_dir)

            data.y_dict = {}
            if "classification" in self.task_type:
                for nodetype, node_label in node_label_dict.items():
                    if np.isnan(node_label).any():
                        data.y_dict[nodetype] = torch.from_numpy(node_label).to(
                            torch.float32
                        )
                    else:
                        data.y_dict[nodetype] = torch.from_numpy(node_label).to(
                            torch.long
                        )
            else:
                for nodetype, node_label in node_label_dict.items():
                    data.y_dict[nodetype] = torch.from_numpy(node_label).to(
                        torch.float32
                    )

        else:
            data = read_graph_pyg(
                self.raw_dir,
                add_inverse_edge=add_inverse_edge,
                additional_node_files=additional_node_files,
                additional_edge_files=additional_edge_files,
                binary=self.binary,
            )[0]

            if self.binary:
                node_label = np.load(osp.join(self.raw_dir, "node-label.npz"))[
                    "node_label"
                ]
            else:
                node_label = pd.read_csv(
                    osp.join(self.raw_dir, "node-label.csv.gz"),
                    compression="gzip",
                    header=None,
                ).values

            if "classification" in self.task_type:
                if np.isnan(node_label).any():
                    data.y = torch.from_numpy(node_label).to(torch.float32)
                else:
                    data.y = torch.from_numpy(node_label).to(torch.long)
            else:
                data.y = torch.from_numpy(node_label).to(torch.float32)

        data = data if self.pre_transform is None else self.pre_transform(data)

        print("Saving...")
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
