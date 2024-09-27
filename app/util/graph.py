import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdmolops
import torch_geometric.utils.smiles as smiles


def smiles2graph(smiles_str):
    data = smiles.from_smiles(smiles_str)
    data.edge_attr = data.edge_attr.float()
    data.x = data.x.float()
    return data

def smiles2graph_arr(arr):
    data = smiles.from_smiles(arr['SMILES'])
    data.edge_attr = data.edge_attr.float()
    data.x = data.x.float()
    data.caption = arr['description']
    return data

# Helper function to one-hot encode atom types
def atom_features(atom):
    atom_type = one_hot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'B', 'H'])
    degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    total_h = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    implicit_valence = one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4])
    aromatic = [1 if atom.GetIsAromatic() else 0]
    return atom_type + degree + total_h + implicit_valence + aromatic

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [1 if x == s else 0 for s in allowable_set]

# Extract features from a molecule
def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Node features
    node_features = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edge index and edge features
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        
        bond_type = bond.GetBondType()
        edge_feature = [1 if bond_type == Chem.rdchem.BondType.SINGLE else 0,
                        1 if bond_type == Chem.rdchem.BondType.DOUBLE else 0,
                        1 if bond_type == Chem.rdchem.BondType.TRIPLE else 0,
                        1 if bond_type == Chem.rdchem.BondType.AROMATIC else 0]
        
        edge_attr.append(edge_feature)
        edge_attr.append(edge_feature)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)