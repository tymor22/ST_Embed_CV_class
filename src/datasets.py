import os
import csv
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from PIL import Image

import pandas as pd
import linecache
import re

from sklearn.metrics import pairwise_distances


###############################################################################################
# iso_cutoff - if set to a nonzero value, drop all spots without a neighbor in that radius.

class STDataset(Dataset):
    def __init__(self, metafile, sep="\t", countfile_format="splotch", preprocess_img=None,
        iso_cutoff=None):

        super(STDataset, self).__init__()

        if not countfile_format == "splotch":
            raise NotImplementedError()

        self.metadata = pd.read_csv(metafile, header=0, sep=sep)
        self.annotated = ("Annotation File" in self.metadata.columns)
        self.sep = sep
        self.iso_cutoff = iso_cutoff

        # For each patch i (0 < i < N), parallel lists containing:
        self.sample_inds = []    # Row index of member array in metadata table.
        self.cfile_inds = []     # Column index of spot i in corresponding count file.
        self.patch_files = []    # Path to histology image of spot i.
        self.afile_inds = []     # Column index of spot i in corresponding annotation file.

        for i in range(len(self.metadata)):
            row = self.metadata.iloc[i]

            if self.annotated:
                cinds, pfiles, ainds = self.preprocess_array(
                    row['Patch Directory'], row['Count File'], row['Annotation File'])
                self.afile_inds += ainds
            else:
                cinds, pfiles = self.preprocess_array(
                    row['Patch Directory'], row['Count File'])
            
            self.sample_inds += ([i] * len(cinds))
            self.cfile_inds += cinds
            self.patch_files += pfiles

        # For fancy indexing later on
        self.sample_inds = np.array(self.sample_inds)
        self.cfile_inds = np.array(self.cfile_inds)
        self.afile_inds = np.array(self.afile_inds)

        # Any provided preprocessing should accepte a PIL Image and output a Tensor.
        if preprocess_img is None:
            self.preprocess_img = Compose([ToTensor()])
        else:
            self.preprocess_img = preprocess_img

    def __len__(self):
        return len(self.sample_inds)

    def __getitem__(self, idx):
        img = Image.open(self.patch_files[idx])
        img = self.preprocess_img(img)

        row = self.metadata.iloc[self.sample_inds[idx]]

        count_vec = pd.read_csv(row['Count File'], header=0, sep=self.sep, 
            usecols=[self.cfile_inds[idx]])
        count_vec = torch.squeeze(torch.tensor(count_vec.values))

        if self.annotated:
            annot_vec = pd.read_csv(row['Annotation File'], header=0, sep=self.sep, 
                usecols=[self.afile_inds[idx]])
            annot_vec = torch.tensor(annot_vec.values)
            return img, count_vec, torch.argmax(annot_vec)

        return img, count_vec        

    def get_neighbors(self, idx, k=1, ccdist=1.2):
        return self.get_distant(idx, k, -ccdist)

    # Returns k patches randomly sampled from 
    def get_distant(self, idx, k=4, ccdist=3.0):
        row = self.metadata.iloc[self.sample_inds[idx]]
        #print(row['Count File'])

        # Find set of all spot coordinates within the current tissue
        in_tissue = (self.sample_inds==self.sample_inds[idx])

        coords = self.coordinates_in(row['Count File'])
        curr = coords[self.cfile_inds[idx] - 1]
        coords = coords[np.array(self.cfile_inds[in_tissue]) - 1]
        
        # Filter spots by desired radius and subsaample
        dmat = pairwise_distances(np.array([curr]), coords)

        if ccdist > 0:
            inds_far = np.where(dmat[0] > ccdist)[0]
        elif ccdist < 0:
            inds_far = np.where((dmat[0] < -ccdist) * (dmat[0] > 0))[0]
        else:
            raise ValueError("ccdist must be non-zero")
        
        if len(inds_far) < k:
            raise ValueError("Insufficient spots meeting distance criterion for point (%.2f,%.2f) in %s " % (curr[0], curr[1], row['Count File']))

        sample_far = np.random.choice(inds_far, size=k, replace=False)

        #print(curr)
        #print(coords[sample_far])

        # Get histology data for subsampled spots
        patches_far = [self.coords_to_spotfile(c) for c in coords[sample_far]]
        patches_far = [os.path.join(row['Patch Directory'], p) for p in patches_far]
        patches_far = [self.preprocess_img(Image.open(f)) for f in patches_far]
        patches_far = torch.cat([torch.unsqueeze(im, 0) for im in patches_far])

        # Get count data for subsampled patches
        count_cols_far = self.cfile_inds[in_tissue][sample_far]
        # In order to preserve sample order, must provide read_csv with column names, not indices...
        col_names = linecache.getline(row['Count File'], 1).rstrip().split(self.sep)
        count_cols_far = np.array(col_names)[count_cols_far]
        counts_far = pd.read_csv(row['Count File'], header=0, sep=self.sep,
            usecols=count_cols_far)[count_cols_far]
        #print(counts_far)
        counts_far = torch.transpose(torch.tensor(counts_far.values), 0, 1)

        # Get aannotation data for subsampled patches (if applicable)
        if self.annotated:
            annot_cols_far = self.afile_inds[in_tissue][sample_far]
            col_names = linecache.getline(row['Annotation File'], 1).rstrip().split(self.sep)
            annot_cols_far = np.array(col_names)[annot_cols_far]
            annots_far = pd.read_csv(row['Annotation File'], header=0, sep=self.sep,
                usecols=annot_cols_far)[annot_cols_far]
            #print(annots_far)
            annots_far = torch.transpose(torch.tensor(annots_far.values), 0, 1) 

            return patches_far, counts_far, np.argmax(annots_far, 1)

        return patches_far, counts_far       


    ##### Helper Functions #####

    # Return (n,2) array of all spot coordinates in count/annotation file header.
    def coordinates_in(self, file):
        header = linecache.getline(file, 1).rstrip()
        cstr_arr = header.split(self.sep)
        c_arr = []
        for i, cstr in enumerate(cstr_arr):
            c = self.parse_coords(cstr)
            
            if c is not None:
                c_arr.append(c)
            elif i != 0 and i != len(cstr_arr)-1:
                raise ValueError("Spot coordinates incorrectly formatted in column %d of %s" 
                    % (i, file))
        return np.array(c_arr)

    # Return (x,y) coordinates from string representation ("x_y")
    def parse_coords(self, cstring):
        tokens = cstring.split('_')
        if len(tokens) != 2:
            return None
        x,y = map(float, tokens)
        return [x,y]

    # File name for spot at location (x,y)
    def coords_to_spotfile(self, c):
        return "%d_%d.jpg" % (int(np.rint(c[0])), int(np.rint(c[1])))

    # Returns a list with an entry for each spot in the array that allows for quick lookup of:
    # - Path to spot image file.
    # - Column index in count array (and annotation array, if applicable).
    # - Column indices of neighboring spots in count array (and annotation array, if applicable).
    def preprocess_array(self, patch_dir, count_file, annot_file=None, ccdist=1.2):

        # Compute pairwise distances between spots in count file (for finding neighbors).
        cfile_coords = self.coordinates_in(count_file)

        # If annotation file is provided, find distances between spots in count file
        #   and those in annotation file.
        # This allows finding of matching spots if column order is not preserved.
        if annot_file is not None:
            afile_coords = self.coordinates_in(annot_file)
            afile_dmat = pairwise_distances(cfile_coords, afile_coords)
            annots = pd.read_csv(annot_file, sep=self.sep, header=0, index_col=0)
            is_annotated = np.sum(annots.values, axis=0)

        # For each spot in count file, track relevant information for getitem:
        l_inds_count = []     # Column index of each spot in count file
        l_patch_files = []    # Path to spot image
        l_inds_annot = []     # Column index of each spot in annot file

        for i in range(cfile_coords.shape[0]):

            # Path to image file for current patch,
            patch_file = os.path.join(patch_dir, self.coords_to_spotfile(cfile_coords[i]))
            if not os.path.exists(patch_file):
                continue
                
            # If annotation file provided, find corresponding column indices therein:
            if annot_file is not None:
                # If set, drop all spots without a neighbor in radius iso_cutoff
                if self.iso_cutoff: 
                    nb_inds = (afile_dmat[i,:] < self.iso_cutoff)
                    if sum(is_annotated[nb_inds]) <= 1:
                        continue

                annot_ind = np.isclose(afile_dmat[i,:], 0, atol=1e-5)
                if np.sum(annot_ind) != 1 or not is_annotated[annot_ind]: 
                    continue
                l_inds_annot.append(np.where(annot_ind)[0][0] + 1)

            l_inds_count.append(i + 1)
            l_patch_files.append(patch_file)

        if annot_file is not None:
            return l_inds_count, l_patch_files, l_inds_annot
        else:
            return l_inds_count, l_patch_files



#################################### ST Neighbors dataseet ###################################

# iso_cutoff - if set to a nonzero value, drop all spots without a neighbor in that radius.

class STDataset_triple(Dataset):
    def __init__(self, metafile, sep="\t", countfile_format="splotch", preprocess_img=None,
        iso_cutoff=None):

        super(STDataset_triple, self).__init__()

        if not countfile_format == "splotch":
            raise NotImplementedError()

        self.metadata = pd.read_csv(metafile, header=0, sep=sep)
        self.annotated = ("Annotation File" in self.metadata.columns)
        self.sep = sep
        self.iso_cutoff = iso_cutoff

        # For each patch i (0 < i < N), parallel lists containing:
        self.sample_inds = []    # Row index of member array in metadata table.
        self.cfile_inds = []     # Column index of spot i in corresponding count file.
        self.patch_files = []    # Path to histology image of spot i.
        self.afile_inds = []     # Column index of spot i in corresponding annotation file.

        for i in range(len(self.metadata)):
            row = self.metadata.iloc[i]

            if self.annotated:
                cinds, pfiles, ainds = self.preprocess_array(
                    row['Patch Directory'], row['Count File'], row['Annotation File'])
                self.afile_inds += ainds
            else:
                cinds, pfiles = self.preprocess_array(
                    row['Patch Directory'], row['Count File'])
            
            self.sample_inds += ([i] * len(cinds))
            self.cfile_inds += cinds
            self.patch_files += pfiles

        # For fancy indexing later on
        self.sample_inds = np.array(self.sample_inds)
        self.cfile_inds = np.array(self.cfile_inds)
        self.afile_inds = np.array(self.afile_inds)

        # Any provided preprocessing should accepte a PIL Image and output a Tensor.
        if preprocess_img is None:
            self.preprocess_img = Compose([ToTensor()])
        else:
            self.preprocess_img = preprocess_img

    def __len__(self):
        return len(self.sample_inds)

    def __getitem__(self, idx):
        img = Image.open(self.patch_files[idx])
        img = self.preprocess_img(img)

        row = self.metadata.iloc[self.sample_inds[idx]]

        count_vec = pd.read_csv(row['Count File'], header=0, sep=self.sep, 
            usecols=[self.cfile_inds[idx]])
        count_vec = torch.squeeze(torch.tensor(count_vec.values))


        k_distant = self.get_distant(idx, k=1, ccdist=3.0)
        k_near = self.get_neighbors(idx, k=1, ccdist=1.2)

        if self.annotated:
            annot_vec = pd.read_csv(row['Annotation File'], header=0, sep=self.sep,
                usecols=[self.afile_inds[idx]])
            annot_vec = torch.tensor(annot_vec.values)
            return img, count_vec, torch.argmax(annot_vec), k_distant, k_near

        return img, count_vec, k_distant, k_near 

    def get_neighbors(self, idx, k=1, ccdist=1.2):
        return self.get_distant(idx, k, -ccdist)

    # Returns k patches randomly sampled from 
    def get_distant(self, idx, k=4, ccdist=3.0):
        row = self.metadata.iloc[self.sample_inds[idx]]
        #print(row['Count File'])

        # Find set of all spot coordinates within the current tissue
        in_tissue = (self.sample_inds==self.sample_inds[idx])

        coords = self.coordinates_in(row['Count File'])
        curr = coords[self.cfile_inds[idx] - 1]
        coords = coords[np.array(self.cfile_inds[in_tissue]) - 1]
        
        # Filter spots by desired radius and subsaample
        dmat = pairwise_distances(np.array([curr]), coords)

        if ccdist > 0:
            inds_far = np.where(dmat[0] > ccdist)[0]
        elif ccdist < 0:
            inds_far = np.where((dmat[0] < -ccdist) * (dmat[0] > 0))[0]
        else:
            raise ValueError("ccdist must be non-zero")
        
        if len(inds_far) < k:
            raise ValueError("Insufficient spots meeting distance criterion for point (%.2f,%.2f) in %s " % (curr[0], curr[1], row['Count File']))

        sample_far = np.random.choice(inds_far, size=k, replace=False)

        #print(curr)
        #print(coords[sample_far])

        # Get histology data for subsampled spots
        patches_far = [self.coords_to_spotfile(c) for c in coords[sample_far]]
        patches_far = [os.path.join(row['Patch Directory'], p) for p in patches_far]
        patches_far = [self.preprocess_img(Image.open(f)) for f in patches_far]
        patches_far = torch.cat([torch.unsqueeze(im, 0) for im in patches_far])

        # Get count data for subsampled patches
        count_cols_far = self.cfile_inds[in_tissue][sample_far]
        # In order to preserve sample order, must provide read_csv with column names, not indices...
        col_names = linecache.getline(row['Count File'], 1).rstrip().split(self.sep)
        count_cols_far = np.array(col_names)[count_cols_far]
        counts_far = pd.read_csv(row['Count File'], header=0, sep=self.sep,
            usecols=count_cols_far)[count_cols_far]
        #print(counts_far)
        counts_far = torch.transpose(torch.tensor(counts_far.values), 0, 1)

        # Get aannotation data for subsampled patches (if applicable)
        if self.annotated:
            annot_cols_far = self.afile_inds[in_tissue][sample_far]
            col_names = linecache.getline(row['Annotation File'], 1).rstrip().split(self.sep)
            annot_cols_far = np.array(col_names)[annot_cols_far]
            annots_far = pd.read_csv(row['Annotation File'], header=0, sep=self.sep,
                usecols=annot_cols_far)[annot_cols_far]
            #print(annots_far)
            annots_far = torch.transpose(torch.tensor(annots_far.values), 0, 1) 

            return patches_far, counts_far, np.argmax(annots_far, 1)

        return patches_far, counts_far       


    ##### Helper Functions #####

    # Return (n,2) array of all spot coordinates in count/annotation file header.
    def coordinates_in(self, file):
        header = linecache.getline(file, 1).rstrip()
        cstr_arr = header.split(self.sep)
        c_arr = []
        for i, cstr in enumerate(cstr_arr):
            c = self.parse_coords(cstr)
            
            if c is not None:
                c_arr.append(c)
            elif i != 0 and i != len(cstr_arr)-1:
                raise ValueError("Spot coordinates incorrectly formatted in column %d of %s" 
                    % (i, file))
        return np.array(c_arr)

    # Return (x,y) coordinates from string representation ("x_y")
    def parse_coords(self, cstring):
        tokens = cstring.split('_')
        if len(tokens) != 2:
            return None
        x,y = map(float, tokens)
        return [x,y]

    # File name for spot at location (x,y)
    def coords_to_spotfile(self, c):
        return "%d_%d.jpg" % (int(np.rint(c[0])), int(np.rint(c[1])))

    # Returns a list with an entry for each spot in the array that allows for quick lookup of:
    # - Path to spot image file.
    # - Column index in count array (and annotation array, if applicable).
    # - Column indices of neighboring spots in count array (and annotation array, if applicable).
    def preprocess_array(self, patch_dir, count_file, annot_file=None, ccdist=1.2):

        # Compute pairwise distances between spots in count file (for finding neighbors).
        cfile_coords = self.coordinates_in(count_file)

        # If annotation file is provided, find distances between spots in count file
        #   and those in annotation file.
        # This allows finding of matching spots if column order is not preserved.
        if annot_file is not None:
            afile_coords = self.coordinates_in(annot_file)
            afile_dmat = pairwise_distances(cfile_coords, afile_coords)
            annots = pd.read_csv(annot_file, sep=self.sep, header=0, index_col=0)
            is_annotated = np.sum(annots.values, axis=0)

        # For each spot in count file, track relevant information for getitem:
        l_inds_count = []     # Column index of each spot in count file
        l_patch_files = []    # Path to spot image
        l_inds_annot = []     # Column index of each spot in annot file

        for i in range(cfile_coords.shape[0]):

            # Path to image file for current patch,
            patch_file = os.path.join(patch_dir, self.coords_to_spotfile(cfile_coords[i]))
            if not os.path.exists(patch_file):
                continue
                
            # If annotation file provided, find corresponding column indices therein:
            if annot_file is not None:
                # If set, drop all spots without a neighbor in radius iso_cutoff
                if self.iso_cutoff: 
                    nb_inds = (afile_dmat[i,:] < self.iso_cutoff)
                    if sum(is_annotated[nb_inds]) <= 1:
                        continue

                annot_ind = np.isclose(afile_dmat[i,:], 0, atol=1e-5)
                if np.sum(annot_ind) != 1 or not is_annotated[annot_ind]: 
                    continue
                l_inds_annot.append(np.where(annot_ind)[0][0] + 1)

            l_inds_count.append(i + 1)
            l_patch_files.append(patch_file)

        if annot_file is not None:
            return l_inds_count, l_patch_files, l_inds_annot
        else:
            return l_inds_count, l_patch_files


class STDataset_no_neighbhors(Dataset):
    def __init__(self, metafile, sep="\t", countfile_format="splotch", preprocess_img=None):
        super(STDataset_no_neighbhors, self).__init__()

        if not countfile_format == "splotch":
            raise NotImplementedError()

        self.metadata = pd.read_csv(metafile, header=0, sep=sep)
        self.annotated = ("Annotation File" in self.metadata.columns)
        self.sep = sep
        self.preprocess_img=preprocess_img
        
        self.patch_list = []
        self.coord_list = []
        self.slide_list = []

        for i in range(len(self.metadata)):
            row = self.metadata.iloc[i]

            count_header = linecache.getline(row['Count File'], 1).rstrip()
            spot_coords = count_header.split(sep)

            # TODO: assert that index column for all count files matches?
            # If the number of genes does not match, trying to batch with a DataLoader will fail.
            # Making sure names/ordering of genes match would avoid insidious errors, but
            #  will cost more preprocessing time.

            # Each entry in the header should be formatted as Xcoord_Ycoord.
            if len(spot_coords) < 2:
                continue

            # If annotations are provided, only include annotated spots.
            if self.annotated:
                annot_header = linecache.getline(row['Annotation File'], 1).rstrip()
                annotated = annot_header.split(sep)
            else:
                annotated = spot_coords

            # Find all spots with both count and image data available
            for spot in spot_coords:
                if spot in annotated:
                    tokens = spot.split('_')
                    if len(tokens) != 2:
                        continue
                    x,y = map(float, tokens)
                    spotfile = "%d_%d.jpg" % (int(np.rint(x)), int(np.rint(y)))

                    if spotfile in os.listdir(row['Patch Directory']):
                        self.patch_list.append(os.path.join(row['Patch Directory'], spotfile))
                        self.coord_list.append(spot)
                        self.slide_list.append(i)

        # Any provided preprocessing should accepte a PIL Image and output a Tensor.
        if preprocess_img is None:
            self.preprocess_img = Compose([ToTensor()])
        else:
            self.preprocess_img = preprocess_img

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        img = Image.open(self.patch_list[idx])
        img = self.preprocess_img(img)

        row = self.metadata.iloc[self.slide_list[idx]]

        count_vec = pd.read_csv(row['Count File'], header=0, sep=self.sep, usecols=[self.coord_list[idx]])
        count_vec = torch.tensor(count_vec.values)

        if self.annotated:
            annot_vec = pd.read_csv(row['Annotation File'], header=0, sep=self.sep, usecols=[self.coord_list[idx]])
            annot_vec = torch.tensor(annot_vec.values)
            return img, count_vec, torch.argmax(annot_vec)

        return img, count_vec

    # TODO: Get neighbors of a patch given an index
    # Will need a way to access neighbors of a given patch within training loop, given:
    # - Spots within a batch may have variable number of neighbors
    # - Might need __getitem__ to return patch index so we can iterate through member patches.
    def get_neighbors(self, idx):
        raise NotImplementedError()
        
        

##################################### DEPRECATED ############################################

''' Unified framework that generates either spot- or grid-level datasets on image patches
    or UMI count data.
    - img_dir - path to directory containing sub-directories for each ST slide.
      Each such directory contains a separate image file for each spot, named according to
      x_y.jpg, where x and y are the integer indices into the ST array.
    - label_dir - path to directory containing PNG files of dimension (H_ST, W_ST), where
      each pixel is a class index between [0, N_class] (0 indicates background).
    - count_dir - path to a directory containing CSV files, where each row is structured
      <x, y, gene_1, ..., gene_G>.

    Naming of all spot subdirectories, label files, and count files must be identical except
    for file extension.
'''

class PatchDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        super(PatchDataset, self).__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir

        self.patch_list = []
        self.coord_list = []

        rxp = re.compile("(\d+)_(\d+).jpg")

        # Look at each sub-directory, which each indicate a separate slide.
        dir_iter = os.walk(img_dir, followlinks=True)
        top = next(dir_iter)
        for root, _, files in dir_iter:

            # Look for all correctly formatted image files within the subdirectories.
            for f in files:
                res = rxp.match(f)
                
                if res is not None:
                    self.patch_list.append(os.path.join(root, f))
                    x, y = int(res.groups()[0]), int(res.groups()[1])
                    self.coord_list.append([os.path.basename(root), x, y])

        if transforms is None:
            self.preprocess = Compose([ToTensor()])
        else:
            self.preprocess = transforms

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        img = Image.open(self.patch_list[idx])
        img = self.preprocess(img)

        base, x, y = self.coord_list[idx]
        lbl = Image.open(os.path.join(self.label_dir, base+".png")).getpixel((x,y))

        return img.float(), torch.tensor(lbl).long()

class CountDataset(Dataset):
    def __init__(self, count_dir, label_dir, normalize_counts=False):
        super(CountDataset, self).__init__()
        self.count_dir = count_dir
        self.label_dir = label_dir
        self.normalize_counts = normalize_counts

        self.spot_inds = []

        def linecount(fname):
            with open(fname) as f:
                for i, l in enumerate(f):
                    pass
            return i  # First line is header

        for f in os.listdir(count_dir):
            if f.endswith(".csv"):
                si = [p for p in enumerate([f.split(".")[0]] * linecount(os.path.join(count_dir,f)))]
                self.spot_inds += si

    def __len__(self):
        return len(self.spot_inds)

    def __getitem__(self, idx):
        line_no, file = self.spot_inds[idx]

        line = linecache.getline(os.path.join(self.count_dir, file+".csv"), line_no+2)
        tokens = line.split(",")
        x, y = int(np.rint(float(tokens[0]))), int(np.rint(float(tokens[1])))
        expr_vec = np.array([float(s) for s in tokens[2:]])

        if self.normalize_counts:
            expr_vec = expr_vec/np.sum(expr_vec)
        
        labels = np.array(Image.open(os.path.join(self.label_dir, file+".png")))

        return torch.from_numpy(expr_vec).float(), torch.tensor(labels[y,x])


import time
from torch.utils.data import DataLoader

if __name__ == "__main__":
    img_dir = "../data/mouse_sc_ccast_pdim280/imgs128/"
    lbl_dir = "../data/mouse_sc_ccast_pdim280/lbls128/"
    count_dir = os.path.expanduser("~/Documents/Splotch_projects/science_rerun/count_matrices_mouse_0305/")
    annot_dir = os.path.expanduser("~/Documents/Splotch_projects/science_rerun/annotations_splotch/")

    metafile = "../data/metadata_test.tsv"
    '''fh = open(metafile, "w+")
    fh.write('\t'.join(["Patch Directory", "Count File", "Annotation File"]))
    fh.write('\n')
    for cfile in os.listdir(count_dir):
        if cfile.endswith(".unified.tsv"):
            tokens = cfile.split("_")
            slide = "_".join(tokens[:2])
            if slide in os.listdir(img_dir) and "%s.tsv" % slide in os.listdir(annot_dir):
                fh.write('\t'.join([
                    os.path.join(img_dir, "%s/" % slide),
                    os.path.join(count_dir, cfile),
                    os.path.join(annot_dir, "%s.tsv" % slide)
                    ]))
                fh.write('\n')
    fh.close()'''

    start = time.time()
    dat1 = STDataset(metafile, precompute_neighbors=False)
    print(len(dat1))
    #dl1 = DataLoader(dat1, batch_size=32, shuffle=True)
    print("STDataset: %d seconds preprocessing" % (time.time()-start))

    idx = 100
    inputs = dat1[idx]
    print("Sample Ind.:", dat1.sample_inds[idx])
    print("Count File Ind:", dat1.cfile_inds[idx])
    print("Annot File Ind:", dat1.afile_inds[idx])
    print("Patch File:", dat1.patch_files[idx])
    print(inputs[0].shape)
    print(torch.max(inputs[1], 0))
    print(inputs[2])

    try:
        nb = dat1.get_neighbors(idx)
        print("Neighbor Patch Files:", dat1.patch_files_nb[idx])
        print(nb[0].shape)
        print(nb[1].shape)
        print(nb[2].shape)
    except NotImplementedError:
        print("Neighborhood calculation disabled")

    '''start = time.time()
    dat2 = PatchDataset(img_dir, lbl_dir)
    dl2 = DataLoader(dat2, batch_size=32, shuffle=True)
    print("PatchDataset: %d seconds preprocessing" % (time.time()-start))

    for dl in [dl1, dl2]:
        start = time.time()
        count = 0
        for batch in dl1:
            count += 1
            if count > 10:
                break
        print(time.time()-start)'''

