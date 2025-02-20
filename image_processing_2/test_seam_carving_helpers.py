#!/usr/bin/env python3

import os
import lab
import sys
import types
import pickle
import hashlib
import collections

import pytest

TEST_DIRECTORY = os.path.dirname(__file__)
sys.path.insert(0, TEST_DIRECTORY)

from test import object_hash, compare_greyscale_images, compare_color_images, load_greyscale_image, qx


def test_pattern_greyscale():
    inpfile = os.path.join(TEST_DIRECTORY, 'test_images', 'pattern.png')
    im = lab.load_color_image(inpfile)
    oim = object_hash(im)
    result = lab.greyscale_image_from_color_image(im)
    expected = {'height': 4,
          'width': 9,
          'pixels': [200, 160, 160, 160, 153, 160, 160, 160, 200,
                     200, 160, 160, 160, 153, 160, 160, 160, 200,
                       0, 153, 160, 160, 160, 160, 160, 153,  0,
                       0, 153, 153, 160, 160, 160, 153, 153,  0]}
    assert object_hash(im)==oim, 'Be careful not to modify the original image!'
    compare_greyscale_images(result, expected)


def test_greyscale():
    for fname in ('centered_pixel', 'smallfrog', 'bluegill', 'twocats', 'tree'):
        inpfile = os.path.join(TEST_DIRECTORY, 'test_images', f'{fname}.png')
        im = lab.load_color_image(inpfile)
        oim = object_hash(im)

        grey = lab.greyscale_image_from_color_image(im)
        expfile = os.path.join(TEST_DIRECTORY, 'test_results', f'{fname}_grey.png')
        assert object_hash(im) == oim, 'Be careful not to modify the original image!'
        compare_greyscale_images(grey, load_greyscale_image(expfile))


def test_pattern_energy():
    im = {'height': 4,
          'width': 9,
          'pixels': [200, 160, 160, 160, 153, 160, 160, 160, 200,
                     200, 160, 160, 160, 153, 160, 160, 160, 200,
                       0, 153, 160, 160, 160, 160, 160, 153,  0,
                       0, 153, 153, 160, 160, 160, 153, 153,  0]}
    oim = object_hash(im)
    result = lab.compute_energy(im)
    expected = {
        'width': 9,
        'height': 4,
        'pixels': [160, 160,  0, 28,  0, 28,  0, 160, 160,
                   255, 218, 10, 22, 14, 22, 10, 218, 255,
                   255, 255, 30,  0, 14,  0, 30, 255, 255,
                   255, 255, 31, 22,  0, 22, 31, 255, 255]
    }
    assert object_hash(im)==oim, 'Be careful not to modify the original image!'
    compare_greyscale_images(result, expected)


def test_energy():
    for fname in ('centered_pixel', 'smallfrog', 'bluegill', 'twocats', 'tree'):
        inpfile = os.path.join(TEST_DIRECTORY, 'test_images', f'{fname}.png')
        im = load_greyscale_image(inpfile)
        oim = object_hash(im)
        result = lab.compute_energy(im)
        assert object_hash(im) == oim, 'Be careful not to modify the original image!'

        expfile = os.path.join(TEST_DIRECTORY, 'test_results', f'{fname}_energy.pickle')
        with open(expfile, 'rb') as f:
            energy = pickle.load(f)

        compare_greyscale_images(result, energy)


def test_pattern_cumulative_energy():
    energy = {
        'width': 9,
        'height': 4,
        'pixels': [160, 160,  0, 28,  0, 28,  0, 160, 160,
                   255, 218, 10, 22, 14, 22, 10, 218, 255,
                   255, 255, 30,  0, 14,  0, 30, 255, 255,
                   255, 255, 31, 22,  0, 22, 31, 255, 255]
    }
    oim = object_hash(energy)
    result = lab.cumulative_energy_map(energy)
    expected = {
        'width': 9,
        'height': 4,
        'pixels': [160, 160,  0, 28,  0, 28,  0, 160, 160,
                   415, 218, 10, 22, 14, 22, 10, 218, 415,
                   473, 265, 40, 10, 28, 10, 40, 265, 473,
                   520, 295, 41, 32, 10, 32, 41, 295, 520]
    }
    assert object_hash(energy)==oim, 'Be careful not to modify the original energy!'
    compare_greyscale_images(result, expected)


def test_cumulative_energy():
    for fname in ('centered_pixel', 'smallfrog', 'bluegill', 'twocats', 'tree'):
        infile = os.path.join(TEST_DIRECTORY, 'test_results', f'{fname}_energy.pickle')
        with open(infile, 'rb') as f:
            energy = pickle.load(f)
        result = lab.cumulative_energy_map(energy)

        expfile = os.path.join(TEST_DIRECTORY, 'test_results', f'{fname}_cumulative_energy.pickle')
        with open(expfile, 'rb') as f:
            cem = pickle.load(f)

        compare_greyscale_images(result, cem)

def check_seam(cem, result, expected):
    assert len(result) == len(expected), f'Expected {len(expected)} indices in seam, but got {len(result)}'
    res_ind = set(result)
    exp_ind = set(expected)
    bad_res_vals = res_ind - exp_ind
    missing_exp_vals = exp_ind - res_ind
    msg = "Correct seam!"
    if missing_exp_vals:
        ind = max(missing_exp_vals)
        row, col = ind // cem['width'], ind % cem['width']
        msg = f'Missing seam index{"es"*(len(missing_exp_vals) > 1)}!'
        msg += f'{qx}Expected index {ind} (row {row}, col {col}) to be present in seam.'
        msg += f'{qx}Found {len(bad_res_vals)} unexpected seam indices in result.'
        res_show = sorted(result)[-10:]
        exp_show = sorted(expected)[-10:]
        if res_show != exp_show:
            msg += f'{qx}Result included  {res_show}'
            msg += f'{qx}but expected had {exp_show}'
    correct_seam = res_ind == exp_ind
    assert correct_seam, msg

def test_pattern_seam_indices():
    cem = {
        'width': 9,
        'height': 4,
        'pixels': [160, 160,  0, 28,  0, 28,  0, 160, 160,
                   415, 218, 10, 22, 14, 22, 10, 218, 415,
                   473, 265, 40, 10, 28, 10, 40, 265, 473,
                   520, 295, 41, 32, 10, 32, 41, 295, 520]
    }
    oim = object_hash(cem)
    result = lab.minimum_energy_seam(cem)
    expected = [2, 11, 21, 31]
    assert object_hash(cem)==oim, 'Be careful not to modify the original cumulative energy map!'
    check_seam(cem, result, expected)


def test_edge_seam_indices():
    cem1 = {
        'width': 5,
        'height': 4,
        'pixels': [20, 20, 20, 20, 20,
                   20, 20, 20, 20, 20,
                   20, 20, 20, 20, 20,
                   20, 20, 20, 20, 20,]
    }
    expected1 = [15, 10, 5, 0] # leftmost column
    result1 = lab.minimum_energy_seam(cem1)
    check_seam(cem1, result1, expected1)

    cem2 = {
        'width': 5,
        'height': 4,
        'pixels': [0,  20, 10, 10, 10,
                   0,  20, 10, 10, 10,
                   0,  20, 10, 10, 10,
                   20, 20, 10, 10, 10,]
    }
    expected2 = [17, 12, 7, 2] # middle column
    result2 = lab.minimum_energy_seam(cem2)
    check_seam(cem2, result2, expected2)

    cem3 = {
        'width': 5,
        'height': 4,
        'pixels': [0,  10,  0, 20, 10,
                   0,  10,  0, 20, 10,
                   0,  10,  0, 20, 10,
                   20, 20, 20, 20, 10,]
    }
    expected3 = [19, 14, 9, 4] # rightmost column
    result3 = lab.minimum_energy_seam(cem3)
    check_seam(cem3, result3, expected3)

    cem4 = {
        'width': 5,
        'height': 4,
        'pixels': [0,  10,  0,  0,  0,
                   0,  10, 20, 10, 10,
                   0,  10, 20, 20, 10,
                   20, 20, 20, 10, 10,]
    }
    expected4 = [18, 14, 8, 2] # correctly choose left in case of ties
    result4 = lab.minimum_energy_seam(cem4)
    check_seam(cem4, result4, expected4)

    cem5 = {
        'width': 5,
        'height': 4,
        'pixels': [10, 20, 20, 20, 0,
                   10, 20, 20, 20, 0,
                   10, 20, 20, 20, 0,
                   10, 20, 20, 20, 10,]
    }
    expected5 = [15, 10, 5, 0] # leftmost column
    result5 = lab.minimum_energy_seam(cem5)
    check_seam(cem5, result5, expected5)


def test_min_seam_indices():
    for fname in ('centered_pixel', 'smallfrog', 'bluegill', 'twocats', 'tree'):
        infile = os.path.join(TEST_DIRECTORY, 'test_results', f'{fname}_cumulative_energy.pickle')
        with open(infile, 'rb') as f:
            cem = pickle.load(f)
        result = lab.minimum_energy_seam(cem)

        expfile = os.path.join(TEST_DIRECTORY, 'test_results', f'{fname}_minimum_energy_seam.pickle')
        with open(expfile, 'rb') as f:
            expected = pickle.load(f)

        check_seam(cem, result, expected)


def test_pattern_seam_removal():
    inpfile = os.path.join(TEST_DIRECTORY, 'test_images', 'pattern.png')
    im = lab.load_color_image(inpfile)
    oim = object_hash(im)
    seam_indices = [2, 11, 21, 31]
    result = lab.image_without_seam(im, seam_indices)
    expfile = os.path.join(TEST_DIRECTORY, 'test_results', 'pattern_1seam.png')
    assert object_hash(im)==oim, 'Be careful not to modify the original image!'
    compare_color_images(result, lab.load_color_image(expfile))


def test_seam_removal():
    for fname in ('pattern', 'bluegill', 'twocats', 'tree'):
        infile = os.path.join(TEST_DIRECTORY, 'test_results', f'{fname}_minimum_energy_seam.pickle')
        with open(infile, 'rb') as f:
            seam = pickle.load(f)

        imfile = os.path.join(TEST_DIRECTORY, 'test_images', f'{fname}.png')
        result = lab.image_without_seam(lab.load_color_image(imfile), seam)

        expfile = os.path.join(TEST_DIRECTORY, 'test_results', f'{fname}_1seam.png')
        compare_color_images(result, lab.load_color_image(expfile))
