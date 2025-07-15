import shutil

import natsort
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, load_json, nifti_files
import SimpleITK as sitk

from nnunetv2.dataset_conversion.Dataset737_convert_to_spheres import generate_segmentation
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

if __name__ == '__main__':
    source_dir = '/media/isensee/raw_data/Dataset704_afids'
    dataset_name = 'Dataset704_afids'
    target_dir = join(nnUNet_raw, dataset_name)

    imagesTr = join(target_dir, 'imagesTr')
    maybe_mkdir_p(imagesTr)
    imagesTs = join(target_dir, 'imagesTs')
    maybe_mkdir_p(imagesTs)
    labelsTr = join(target_dir, 'labelsTr')
    maybe_mkdir_p(labelsTr)
    labelsTs = join(target_dir, 'labelsTs')
    maybe_mkdir_p(labelsTs)

    coords = load_json(join(source_dir, 'afids_landmark_coordinates.json'))
    all_landmarks = list(set([i for j in coords.keys() for i in coords[j].keys()]))
    all_landmarks = natsort.natsorted(all_landmarks)

    train_identifiers = [i[:-12] for i in nifti_files(join(source_dir, 'imagesTr'), join=False)]
    test_identifiers = [i[:-12] for i in nifti_files(join(source_dir, 'imagesTs'), join=False)]
    all_coords = {}

    for k in coords.keys():
        is_train = k in train_identifiers

        img = join(source_dir, 'imagesTr' if is_train else 'imagesTs', k + '_0000.nii.gz')
        img_itk = sitk.ReadImage(img)
        shape = sitk.GetArrayFromImage(img_itk).shape

        coords_here = {int(ky): np.round(coords[k][ky])[::-1] for ky in all_landmarks}

        all_coords[k] = coords_here

        seg = generate_segmentation(shape, coords_here, radius=2)
        seg_itk = sitk.GetImageFromArray(seg)
        seg_itk.SetSpacing(img_itk.GetSpacing())
        seg_itk.SetOrigin(img_itk.GetOrigin())
        seg_itk.SetDirection(img_itk.GetDirection())

        sitk.WriteImage(seg_itk, join(labelsTr if is_train else labelsTs, k + '.nii.gz'))
        shutil.copy(img, join(imagesTr if is_train else imagesTs, k + '_0000.nii.gz'))

    generate_dataset_json(
        target_dir,
        {0: 'MRI'},
        {'background': 0, **{f"landmark_{int(k):02d}": int(k) for k in all_landmarks}},
         len(train_identifiers),
        '.nii.gz',
        citation='',
        regions_class_order=None,
        dataset_name=dataset_name,
        reference='',
        license='',
        converted_by='Fabian Isensee',
        note='This is a landmark detection dataset.',
    )
