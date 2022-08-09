#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from pathlib import Path
import glob
from src import tools
import nibabel as nib
import seaborn as sns
from nilearn import plotting


class VoxelPermutation:
    def __init__(self, args):
        self.process = 'VoxelPermutation'
        self.model = args.model.replace('_', ' ')
        self.unique_model = args.unique_model
        self.single_model = args.single_model
        if self.unique_model is not None:
            self.unique_model = self.unique_model.replace('_', ' ')
        if self.single_model is not None:
            self.single_model = self.single_model.replace('_', ' ')
        self.sid = str(args.s_num).zfill(2)
        self.cross_validation = args.CV
        if self.cross_validation:
            self.method = 'CV'
        else:
            self.method = 'test'
        self.n_perm = args.n_perm
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.figure_dir = f'{args.figure_dir}/{self.process}'
        Path(f'{self.out_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(f'{self.out_dir}/{self.process}/rnull').mkdir(parents=True, exist_ok=True)
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)
        instance_variables = vars(self)
        print(instance_variables)

    def load_unique(self):
        if self.cross_validation:
            fnames_loo = f'{self.out_dir}/VoxelRegression/sub-{self.sid}_prediction-all_drop-{self.unique_model}_single-None_method-CV_loop*.npy'
            fnames_all = f'{self.out_dir}/VoxelRegression/sub-{self.sid}_prediction-all_drop-None_single-None_method-CV_loop*.npy'
            full_files = sorted(glob.glob(fnames_all))
            print(full_files)
            loo_files = sorted(glob.glob(fnames_loo))
            test_files = sorted(glob.glob(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_y-test_method-CV_loop*.npy'))
            loo = None
            for i, ((full_file, loo_file), test_file) in enumerate(zip(zip(full_files, loo_files), test_files)):
                full_file = np.load(full_file)
                loo_file = np.load(loo_file)
                test_file = np.load(test_file)
                if loo is None:
                    loo = np.zeros((loo_file.shape[0], len(loo_file), loo_file.shape[1]))
                    full = np.zeros_like(loo)
                    test = np.zeros_like(loo)
                loo[:, i, :] = loo_file
                full[:, i, :] = full_file
                test[:, i, :] = test_file
        else:
            full = np.load(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_prediction-all_drop-None_single-None_method-test.npy')
            loo = np.load(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_prediction-all_drop-{self.unique_model}single-None_method-test.npy')
            test = np.load(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_y-test_method-test.npy')
        return full, loo, test

    def load(self):
        if self.cross_validation:
            fnames = f'{self.out_dir}/VoxelRegression/sub-{self.sid}_prediction-{self.model}_drop-{self.unique_model}_single-{self.single_model}_method-CV_loop*.npy'
            print(fnames)
            pred_files = sorted(glob.glob(fnames))
            test_files = sorted(
                glob.glob(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_y-test_method-CV_loop*.npy'))
            pred = None
            for i, (pred_file, test_file) in enumerate(zip(pred_files, test_files)):
                pred_file = np.load(pred_file)
                test_file = np.load(test_file)
                if pred is None:
                    pred = np.zeros((pred_file.shape[0], len(pred_files), pred_file.shape[1]))
                    test = np.zeros_like(pred)
                pred[:, i, :] = pred_file
                test[:, i, :] = test_file
        else:
            pred = np.load(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_prediction-{self.model}_drop-{self.unique_model}_single-{self.single_model}_method-test.npy')
            test = np.load(f'{self.out_dir}/VoxelRegression/sub-{self.sid}_y-test_method-test.npy')
        return test, pred

    def load_anatomy(self):
        anat = nib.load(f'{self.data_dir}/anatomy/sub-{self.sid}/sub-{self.sid}_desc-preproc_T1w.nii.gz')
        brain_mask = nib.load(f'{self.data_dir}/anatomy/sub-{self.sid}/sub-{self.sid}_desc-brain_mask.nii.gz')
        return tools.mask_img(anat, brain_mask)

    def nib_transform(self, r_, nii=True):
        betas = nib.load(f'{self.data_dir}/betas/sub-{self.sid}/sub-{self.sid}_space-T1w_desc-train-fracridge_data.nii.gz')
        unmask = np.load(
            f'{self.out_dir}/Reliability/sub-{self.sid}_space-T1w_desc-test-fracridge_reliability-mask.npy').astype(
            'bool')
        i = np.where(unmask)
        if r_.ndim < 2:
            r_unmasked = np.zeros(unmask.shape)
            r_unmasked[i] = r_
            r_unmasked = r_unmasked.reshape(betas.shape[:-1])
        else:
            r_ = r_.T
            r_unmasked = np.zeros((unmask.shape + (r_.shape[-1],)))
            r_unmasked[i, ...] = r_
            r_unmasked = r_unmasked.reshape((betas.shape[:-1] + (r_.shape[-1],)))
            print(r_unmasked.shape)

        if nii:
            r_unmaksed = nib.Nifti1Image(r_unmasked, betas.affine, betas.header)
        return r_unmaksed

    def plot_results(self, r_):
        print('plotting results')
        anatomy = self.load_anatomy()
        figure_name = f'{self.figure_dir}/sub-{self.sid}_prediction-{self.model}_drop-{self.unique_model}_single-{self.single_model}_method-{self.method}.png'
        plotting.plot_stat_map(r_, anatomy,
                               symmetric_cbar=False,
                               threshold=1e-6,
                               display_mode='mosaic',
                               cmap=sns.color_palette('magma', as_cmap=True),
                               output_file=figure_name)

    def save_perm_results(self, d):
        print('Saving output')
        base = f'{self.out_dir}/{self.process}/sub-{self.sid}_prediction-{self.model}_drop-{self.unique_model}_single-{self.single_model}_method-{self.method}'
        for key in d.keys():
            nib.save(d[key], f'{base}_{key}.nii.gz')
        print('Completed successfully!')

    def run(self):
        if self.unique_model is None:
            y_true, y_pred = self.load()
            print(np.unique(y_true))
            r2, p, r2_null = tools.perm(y_true, y_pred, n_perm=self.n_perm)
        else:
            y_full_pred, y_loo_pred, y_true = self.load_unique()
            r2, p, r2_null = tools.perm_unique_variance(y_true, y_full_pred,
                                                           y_loo_pred, n_perm=self.n_perm)

        #filter the rs based on the significant voxels
        r2_filtered, p_corrected = tools.filter_r(r2.copy(), p)

        #transform arrays to nii
        out_data = dict()
        for name, i in zip(['r2', 'r2filtered', 'p', 'pcorrected', 'r2null'],
                           [r2, r2_filtered, p, p_corrected, r2_null]):
            out_data[name] = self.nib_transform(i)
        self.plot_results(out_data['r2filtered'])
        self.save_perm_results(out_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=int, default=1)
    parser.add_argument('--unique_model', type=str, default=None)
    parser.add_argument('--single_model', type=str, default=None)
    parser.add_argument('--model', type=str, default='all')
    parser.add_argument('--CV', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--n_perm', type=int, default=5000)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/raw')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/data/interim')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_analysis/reports/figures')
    args = parser.parse_args()
    VoxelPermutation(args).run()

if __name__ == '__main__':
    main()