import math
import threading
import os
from time import time

import cv2 as cv
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
# pycuda
import pycuda.gpuarray as gpuarray
from scipy import special

from utils import *
import kernels

# import image and k
file_name = 'lena-256.tif'
file_dir = os.path.join('..', '..', 'images')
file_path = 'lena.png'

k = 1
img = cv.imread(file_path, 0)
img = img.astype(np.float64)

# get shape and width
shape = img.shape
width = shape[0]
width_k = width * k
middle = width/2
middle_k = width_k/2
width_special = 8
special_scale = 7

print('\nk: %s' % k)
print('\nspecial scale:: %s' % special_scale)

# img k
img_k = cv.resize(img, (width*k, width*k), interpolation=cv.INTER_NEAREST)
special_start = int(middle - width_special/2)
special_end = int(middle + width_special/2)
width_special_scaled = width_special * special_scale
middle_special_scaled = int(width_special_scaled/2)
overall_k_special_width = width_k + width_special_scaled
img_special = cv.resize(
    img[special_start:special_end, special_start:special_end],
    (width_special_scaled, width_special_scaled),
    interpolation=cv.INTER_NEAREST
)
img_special_flat = img_special.ravel()

# overall timing
overall_timing = time()

# compute radius for k first
radius_k_ndarray = np.array([[get_normalized_radius(
    i+1, j+1, middle_k) for i in range(width_k)] for j in range(width_k)], np.float64)

# then compute radius for special, not need to check radius for now
special_to_full_ratio = width_special / width
radius_special_ndarray = np.array([[get_normalized_radius(
    i+1, j+1, middle_special_scaled) for i in range(width_special_scaled)] for j in range(width_special_scaled)],
    np.float64) * special_to_full_ratio
radius_special_ndarray_flat = radius_special_ndarray.ravel()

print('\n--Get Mask--\n')
timing = time()
# mask for non-k, will be changed shortly
radius_ndarray_mask = np.ones((width, width)).astype(np.float64)
radius_ndarray_mask_for_k = np.ones((width, width)).astype(np.float64)

radius_ndarray = np.array([[get_normalized_radius(i+1, j+1, middle)
                            for i in range(width)] for j in range(width)], np.float64)
for j in range(width_k):
    new_j = int(j/k)
    for i in range(width_k):
        new_i = int(i/k)
        if radius_k_ndarray[i][j] > 1:
            radius_ndarray_mask[new_i][new_j] = 0
            radius_ndarray_mask_for_k[new_i][new_j] = 0
        if new_i >= special_start and new_i < special_end and new_j >= special_start and new_j < special_end:
            radius_ndarray_mask_for_k[new_i][new_j] = 0

print('for-loop time: %s' % (time()-timing))

# get mask for k from non-k mask
radius_k_ndarray_mask = cv.resize(
    radius_ndarray_mask_for_k, (width_k, width_k), interpolation=cv.INTER_NEAREST)

radius_ndarray_flat_masked = (radius_ndarray * radius_ndarray_mask).ravel()
radius_k_ndarray_flat_filtered = radius_k_ndarray[
    np.where(radius_k_ndarray_mask > 0)].ravel()

img_ndarray_flat = img.ravel()
img_ndarray_masked = img * radius_ndarray_mask

img_k_ndarray_flat_filtered = img_k[
    np.where(radius_k_ndarray_mask > 0)].ravel()
print('Get Mask time: %s' % (time()-timing))

# arctan
print('\n--Actan--\n')
timing = time()
arctan_ndarray_flat = np.zeros(width*width, np.float64)
arctan_k_ndarray = np.zeros(width_k*width_k, np.float64)
arctan_special_ndarray_flat = np.zeros(
    width_special_scaled*width_special_scaled, np.float64)

block_width = 32
threads_per_block = (block_width, block_width, 1)
grid_x = int(width_k-1) // block_width + 1
grid_y = int(width_k-1) // block_width + 1
blocks_per_grid = (grid_x, grid_y, 1)

arctan_kernel_mod = kernels.arctan_kernel_mod
arctan_kernel = arctan_kernel_mod.get_function("arctan_kernel")
arctan_kernel(
    drv.Out(arctan_k_ndarray),
    drv.Out(arctan_ndarray_flat),
    drv.Out(arctan_special_ndarray_flat),
    drv.In(np.int32(width)),
    drv.In(np.int32(k)),
    drv.In(np.int32(width_special_scaled)),
    drv.In(np.int32(middle)),
    drv.In(np.int32(middle_k)),
    drv.In(np.int32(middle_special_scaled)),
    block=threads_per_block, grid=blocks_per_grid
)

arctan_k_ndarray_flat_filtered = arctan_k_ndarray[
    np.where(radius_k_ndarray_mask.ravel() > 0)]

print('gpu time: %s' % (time()-timing))

# order and zero_crossings
v = 1.
order = N = 500
M = order * 2 + 1
print('\n--zero crossings and "an" array--')
timing = time()
zero_crossings = special.jn_zeros(v, order).astype(np.float64)
print('time: %s' % (time()-timing))

# kernel definitions
jv_kernel_mod = kernels.jv_kernel_mod

exp_kernel_mod = kernels.exp_kernel_mod

bnm_kernel_mod = kernels.bnm_kernel_mod

# partition flat arrays for computing jv

# size for partition
element_size = 8
# reserve 300 mb for other kernel opeerations
p_size_multiplied_in_mb = 4096-300
size_left = img_k_ndarray_flat_filtered.size * \
    img_k_ndarray_flat_filtered.itemsize
p_size = int((p_size_multiplied_in_mb * 1024 ** 2 - M*N*2*element_size) /
             ((2*M+N+1) * 1024 ** 2)) * 1024 ** 2
loop_count = 0
print('\n---size section---')
print('p_size : %s MB' % (p_size/1024**2))
print('size_left: %s MB' % (size_left/1024**2))
print_divider()
bnm_list = []
init_timing = timing = time()
end_pos = -1

print('\n---Normal part---\n')
while size_left > 0:
    print('#####Loop %s####' % (loop_count + 1))
    # update sizes
    current_size = p_size
    if size_left < p_size:
        current_size = size_left
    size_left -= p_size
    start_pos = end_pos + 1
    end_pos = start_pos + int(current_size/element_size)
    loop_count += 1
    print('current_size: %s MB' % (current_size/1024**2))

    pixels_in_part = end_pos - start_pos
    print('pixels_in_part: %s' % pixels_in_part)
    pixels_total = pixels_in_part * (M+N)

    # jv
    print('\n--jv--')
    pixels_total_n = pixels_in_part * N
    radius_k_ndarray_flat_masked_p = radius_k_ndarray_flat_filtered[start_pos: end_pos]

    jv_normalized_k_ndarray_flat_n_masked_p = np.zeros(
        (N, pixels_in_part), np.float64)

    block_width = 32
    threads_per_block = (block_width, block_width, 1)

    grid_x = int(pixels_total_n-1) // (block_width * block_width) + 1
    blocks_per_grid = (grid_x, 1, 1)
    print('threads per block : (%s, %s, %s)' % threads_per_block)
    print('blocks per grid : %s' % blocks_per_grid[0])
    print('n : %s' % N)
    print('pixels_in_part : %s' % pixels_in_part)
    print('total_pixels : %s' % (pixels_in_part*N))

    jv_kernel = jv_kernel_mod.get_function("jv_kernel")

    (constant_ptr, constant_size) = jv_kernel_mod.get_global('zero_crossings')
    pycuda.driver.memcpy_htod(
        constant_ptr, zero_crossings)

    timing = time()
    kernel_time = jv_kernel(
        drv.Out(jv_normalized_k_ndarray_flat_n_masked_p),
        drv.In(radius_k_ndarray_flat_masked_p),
        drv.In(np.int32(N)),
        drv.In(np.int32(pixels_in_part)),
        block=threads_per_block, grid=blocks_per_grid,
        time_kernel=True)
    print('kernel time : %s' % kernel_time)
    print('time : %s' % (time()-timing))

    # exp
    print('\n--exp--')
    pixels_total_m = pixels_in_part * M
    grid_x = int(pixels_total_m-1) // (block_width * block_width) + 1
    blocks_per_grid = (grid_x, 1, 1)

    exp_kernel = exp_kernel_mod.get_function("exp_kernel")

    arctan_k_ndarray_flat_p = arctan_k_ndarray_flat_filtered[start_pos: end_pos]
    exp_k_ndarray_flat_m_p = np.zeros((M, pixels_in_part), np.complex128)
    timing = time()
    kernel_time = exp_kernel(
        drv.Out(exp_k_ndarray_flat_m_p),
        drv.In(arctan_k_ndarray_flat_p),
        drv.In(np.int32(M)),
        drv.In(np.int32(pixels_in_part)),
        block=threads_per_block, grid=blocks_per_grid,
        time_kernel=True)
    print('kernel time : %s' % kernel_time)
    print('time : %s' % (time()-timing))
    # print(exp_k_ndarray_flat_m_p)

    # bnm
    print('\n--join bnm--')

    bnm_kernel = bnm_kernel_mod.get_function("bnm_kernel")

    grid_x = int(M-1) // block_width + 1
    grid_y = int(N-1) // block_width + 1
    blocks_per_grid = (grid_x, grid_y, 1)
    print('grid config : %s' % blocks_per_grid.__str__())

    img_k_ndarray_flat_p = img_k_ndarray_flat_filtered[start_pos: end_pos]
    bnm_ndarray_p = np.zeros((N, M), np.complex128)
    timing = time()
    kernel_time = bnm_kernel(
        drv.Out(bnm_ndarray_p),
        drv.In(jv_normalized_k_ndarray_flat_n_masked_p),
        drv.In(img_k_ndarray_flat_p),
        drv.In(exp_k_ndarray_flat_m_p),
        drv.In(np.int32(N)),
        drv.In(np.int32(M)),
        drv.In(np.int32(pixels_in_part)),
        block=threads_per_block, grid=blocks_per_grid,
        time_kernel=True)
    print('kernel time : %s' % kernel_time)
    print('bnm time : %s' % (time()-timing))

    bnm_list.append(bnm_ndarray_p)


# bnm for special
print('\n--Special part--\n')

bnm_special_list = []
loop_count = 0
size_left = img_special_flat.size * \
    img_special_flat.itemsize
end_pos = -1
while(size_left > 0):
    print('#####Loop %s####' % (loop_count + 1))
    # update sizes
    current_size = p_size
    if size_left < p_size:
        current_size = size_left
    size_left -= p_size
    start_pos = end_pos + 1
    end_pos = start_pos + int(current_size/element_size)
    loop_count += 1
    print('\ncurrent_size: %s MB' % (current_size/1024**2))

    pixels_in_part = end_pos - start_pos
    print('pixels_in_part: %s' % pixels_in_part)

    # jv
    print('\n--jv--')
    pixels_total_n = pixels_in_part * N
    radius_special_ndarray_flat_p = radius_special_ndarray_flat[start_pos: end_pos]

    jv_normalized_special_ndarray_flat_n_p = np.zeros(
        (N, pixels_in_part), np.float64)

    block_width = 32
    threads_per_block = (block_width, block_width, 1)

    grid_x = int(pixels_total_n-1) // (block_width * block_width) + 1
    blocks_per_grid = (grid_x, 1, 1)
    print('threads per block : (%s, %s, %s)' % threads_per_block)
    print('blocks per grid : %s' % blocks_per_grid[0])
    print('n : %s' % N)
    print('pixels_in_part : %s' % pixels_in_part)
    print('total_pixels : %s' % (pixels_in_part*N))

    jv_kernel = jv_kernel_mod.get_function("jv_kernel")
    (constant_ptr, constant_size) = jv_kernel_mod.get_global('zero_crossings')
    pycuda.driver.memcpy_htod(
        constant_ptr, zero_crossings)
    timing = time()
    kernel_time = jv_kernel(
        drv.Out(jv_normalized_special_ndarray_flat_n_p),
        drv.In(radius_special_ndarray_flat_p),
        drv.In(np.int32(N)),
        drv.In(np.int32(pixels_in_part)),
        block=threads_per_block, grid=blocks_per_grid,
        time_kernel=True)
    print('kernel time : %s' % kernel_time)
    print('time : %s' % (time()-timing))

    # exp
    print('\n--exp--')
    pixels_total_m = pixels_in_part * M
    grid_x = int(pixels_total_m-1) // (block_width * block_width) + 1
    blocks_per_grid = (grid_x, 1, 1)

    arctan_special_ndarray_flat_p = arctan_special_ndarray_flat[start_pos: end_pos]
    exp_special_ndarray_flat_m_p = np.zeros((M, pixels_in_part), np.complex128)

    exp_kernel = exp_kernel_mod.get_function("exp_kernel")
    timing = time()
    kernel_time = exp_kernel(
        drv.Out(exp_special_ndarray_flat_m_p),
        drv.In(arctan_special_ndarray_flat_p),
        drv.In(np.int32(M)),
        drv.In(np.int32(pixels_in_part)),
        block=threads_per_block, grid=blocks_per_grid,
        time_kernel=True)
    print('kernel time : %s' % kernel_time)
    print('time : %s' % (time()-timing))

    # bnm
    print('\n--join bnm--')

    bnm_kernel = bnm_kernel_mod.get_function("bnm_kernel")

    grid_x = int(M-1) // block_width + 1
    grid_y = int(N-1) // block_width + 1
    blocks_per_grid = (grid_x, grid_y, 1)
    print('grid config : %s' % blocks_per_grid.__str__())

    img_special_ndarray_flat_p = img_special_flat[start_pos: end_pos]
    bnm_ndarray_p = np.zeros((N, M), np.complex128)
    timing = time()
    kernel_time = bnm_kernel(
        drv.Out(bnm_ndarray_p),
        drv.In(jv_normalized_special_ndarray_flat_n_p),
        drv.In(img_special_ndarray_flat_p),
        drv.In(exp_special_ndarray_flat_m_p),
        drv.In(np.int32(N)),
        drv.In(np.int32(M)),
        drv.In(np.int32(pixels_in_part)),
        block=threads_per_block, grid=blocks_per_grid,
        time_kernel=True)
    print('kernel time : %s' % kernel_time)
    print('time : %s' % (time()-timing))
    bnm_special_list.append(bnm_ndarray_p)

# EOF special bnm
print('\nTime used for computing bnm: %s' % (time() - init_timing))


print('\nAdding bnm arrays up')
bnm_special_ndarray = np.zeros((N, M), np.complex128)
bnm_ndarray = np.zeros((N, M), np.complex128)
timing = time()
for bnm_p in bnm_list:
    bnm_ndarray += bnm_p
for bnm_p in bnm_special_list:
    bnm_special_ndarray += bnm_p
special_to_k = special_scale**2 / k**2
bnm_multiplier = 2/(np.pi*width**2*k**2)
bnm_ndarray = (bnm_ndarray + bnm_special_ndarray/special_to_k) * bnm_multiplier
print('time : %s' % (time()-timing))


# preparation for reconstruction
print('\nPreparation for reconstruction')

print('\n--jv--')
pixels_total = width * width
pixels_total_n = pixels_total * N
jv_ndarray_flat_n_masked = np.zeros(
    (N, pixels_total), np.float64)

block_width = 32
threads_per_block = (block_width, block_width, 1)

grid_x = int(pixels_total_n-1) // (block_width * block_width) + 1
blocks_per_grid = (grid_x, 1, 1)

jv_non_normalized_kernel_mod = kernels.jv_non_normalized_kernel_mod
jv_non_normalized_kernel = jv_non_normalized_kernel_mod.get_function(
    "jv_non_normalized_kernel")

(constant_ptr, constant_size) = jv_non_normalized_kernel_mod.get_global('zero_crossings')
pycuda.driver.memcpy_htod(
    constant_ptr, zero_crossings)

timing = time()
kernel_time = jv_non_normalized_kernel(
    drv.Out(jv_ndarray_flat_n_masked),
    drv.In(radius_ndarray_flat_masked),
    drv.In(np.int32(N)),
    drv.In(np.int32(pixels_total)),
    block=threads_per_block, grid=blocks_per_grid,
    time_kernel=True)
print('kernel time : %s' % kernel_time)
print('time : %s' % (time()-timing))

print('\n--exp--')
pixels_total_m = pixels_total * M
grid_x = int(pixels_total_m-1) // (block_width * block_width) + 1
blocks_per_grid = (grid_x, 1, 1)

exp_ndarray_flat_m = np.zeros((M, pixels_total), np.complex128)

exp_kernel = exp_kernel_mod.get_function("exp_kernel")
timing = time()
kernel_time = exp_kernel(
    drv.Out(exp_ndarray_flat_m),
    drv.In(arctan_ndarray_flat),
    drv.In(np.int32(M)),
    drv.In(np.int32(pixels_total)),
    block=threads_per_block, grid=blocks_per_grid,
    time_kernel=True)
print('kernel time : %s' % kernel_time)
print('time : %s' % (time()-timing))

# exit()

# reconstruction
print('\n--reconstruction--')
timing = time()
img_out_ndarray = np.real((jv_ndarray_flat_n_masked.T.dot(
    bnm_ndarray)*np.conj(exp_ndarray_flat_m.T)).dot(np.ones(2*order+1)).reshape((width, width)))
print('time : %s' % (time()-timing))

print('\n--Overall Timing--')
print('%s seconds' % (time()-overall_timing))

# PSNR
mse = np.sum((img_out_ndarray - img_ndarray_masked)**2)/width**2
psnr = 10*np.log10(255 ** 2/mse)
print('\n\nPSNR : %s' % psnr)

# write to file
output_file_name = '%s__%s__%s__%s.jpg' % (
    get_filename_no_ext(__file__), get_commit_hash(), get_filename_no_ext(file_name), k)
output_file_path = os.path.join('..', 'output_images', output_file_name)
cv.imwrite('output.jpg', img_out_ndarray)
