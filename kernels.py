# this module contains all cuda kernels to be used

from pycuda.compiler import SourceModule

arctan_kernel_mod = SourceModule("""
__global__ void arctan_kernel(
        double * ar_arctan_k,
        double * ar_arctan,
        double * ar_arctan_special,
        int * width, 
        int * k,
        int * width_special_scaled,
        int * middle,
        int * middle_k,
        int * middle_special_scaled
    )
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bw = blockDim.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int col = bx*bw + tx;
    const int row = by*bw + ty;

    int pos_in_area;
    double angle;

    const int width_k = *width * *k;
    const double pi = atan(1.)*4;

    // load ar_arctan_k
    if(col < width_k && row < width_k){
        pos_in_area = row * width_k + col; 
        angle = atan2(-row-1+ *middle_k+1./2, col+1- *middle_k-1./2);
        ar_arctan_k[pos_in_area] = angle < 0 ? angle + 2*pi : angle;
    }

    // load ar_arctan
    if(col < *width && row < *width){
        pos_in_area = row * *width + col; 
        angle = atan2(-row-1+ *middle+1./2, col+1- *middle-1./2);
        ar_arctan[pos_in_area] = angle < 0 ? angle + 2*pi : angle;
    }

    
    // load ar_arctan_special
    if(col < *width_special_scaled && row < *width_special_scaled){
        pos_in_area = row * *width_special_scaled + col; 
        angle = atan2(-row-1+ *middle_special_scaled+1./2, col+1- *middle_special_scaled-1./2);
        ar_arctan_special[pos_in_area] = angle < 0 ? angle + 2*pi : angle;
    }
    /**
    */
}""")

jv_kernel_mod = SourceModule("""
__constant__ double zero_crossings[2048];
__global__ void jv_kernel(double * ar_c, double * ar_a, int * n, int * pixels_in_part)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bw = blockDim.x;
    const int bh = blockDim.y;
    const int bx = blockIdx.x;
    int pos = tx + ty*bw + bx*bw*bh;

    if(pos < (*n) * (*pixels_in_part)){
        int idx_origin_y = pos / (*pixels_in_part);
        int idx_origin_x = pos - idx_origin_y*(*pixels_in_part);
        ar_c[pos] = j1(ar_a[idx_origin_x] * zero_crossings[idx_origin_y]) / \
                       (pow(jn(2, zero_crossings[idx_origin_y]), 2.)/2);
    }
}""")

jv_non_normalized_kernel_mod = SourceModule("""
__constant__ double zero_crossings[2048];
__global__ void jv_non_normalized_kernel(double * ar_c, double * ar_a, int * n, int * pixels_in_part)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bw = blockDim.x;
    const int bh = blockDim.y;
    const int bx = blockIdx.x;
    int pos = tx + ty*bw + bx*bw*bh;

    if(pos < (*n) * (*pixels_in_part)){
        int idx_origin_y = pos / (*pixels_in_part);
        int idx_origin_x = pos - idx_origin_y*(*pixels_in_part);
        ar_c[pos] = j1(ar_a[idx_origin_x] * zero_crossings[idx_origin_y]);
    }
}""")

exp_kernel_mod = SourceModule("""
    # include <pycuda-complex.hpp>
    typedef pycuda::complex<double> dcmplx;
    __global__ void exp_kernel(dcmplx *ar_exp, double *ar_arctan, int *m, int *pixels_in_part)
    {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int bw = blockDim.x;
        const int bh = blockDim.y;
        const int bx = blockIdx.x;
        int pos = tx + ty*bw + bx*bw*bh;
        if(pos < (*m) * (*pixels_in_part)){
            int idx_origin_y = pos / (*pixels_in_part);
            int idx_origin_x = pos % (*pixels_in_part);
            ar_exp[pos] = exp(
                -dcmplx(
                    0,
                    ar_arctan[idx_origin_x]*(double)(idx_origin_y-(*m-1)/2)
                )
            );
        }
    }""")

bnm_kernel_mod = SourceModule("""
    # include <pycuda-complex.hpp>
    typedef pycuda::complex<double> dcmplx;
    __global__ void bnm_kernel(
        dcmplx *ar_bnm,
        double *ar_jv,
        double *ar_img_k,
        dcmplx *ar_exp,
        int *n,
        int *m,
        int *pixels_in_part
    )
    {
        const int bx = blockIdx.x;
        const int by = blockIdx.y;
        const int bw = blockDim.x;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        const int n_local = *n;
        const int m_local = *m;
        const int pixels_in_part_local = *pixels_in_part;
        const int ph_len = 62;

        // identify the row and col num for the current thread
        const int row = by * bw + ty;
        const int col = bx * bw + tx;
        const int pos = row * m_local + col;

        const int block_pos = tx + bw*ty;

        dcmplx total = dcmplx(0,0);

        // allocate shared memory
        __shared__ double ar_jv_ph[32][ph_len];
        __shared__ dcmplx ar_exp_ph[32][ph_len];
        __shared__ double ar_img_k_ph[ph_len];

        // loop over tiles in ar_jv and ar_exp to calculate

        for (int ph = 0; ph <= (pixels_in_part_local-1)/ph_len; ph++){

            // load data into shared momery && boundary checking
            const int iteLen = (ph+1)*ph_len > pixels_in_part_local ? pixels_in_part_local-ph*ph_len : ph_len;

            const int base_load_count = iteLen / bw;
            const int extra_load_jv_count = tx < iteLen % bw ? 1 : 0;
            const int extra_load_exp_count = ty < iteLen % bw ? 1 : 0;

            if(row < n_local){
                for(int i=0; i<base_load_count; i++){
                    ar_jv_ph[ty][i*bw + tx] = ar_jv[row * \
                        pixels_in_part_local + ph*ph_len + i*bw + tx];
                }
                if(extra_load_jv_count){
                    ar_jv_ph[ty][base_load_count*bw + tx] = ar_jv[row * \
                        pixels_in_part_local + ph*ph_len + base_load_count*bw + tx];
                }
            }

            if(col < m_local){
                for(int i=0; i<base_load_count; i++){
                    ar_exp_ph[tx][i*bw + ty] = ar_exp[col * \
                        pixels_in_part_local + ph*ph_len + i*bw + ty];
                }
                if(extra_load_exp_count){
                    ar_exp_ph[tx][base_load_count*bw + ty] = ar_exp[col * \
                        pixels_in_part_local + ph*ph_len + base_load_count*bw + ty];
                }
            }

            if(block_pos < iteLen){
                ar_img_k_ph[block_pos] = ar_img_k[block_pos + ph_len * ph];
            }

            __syncthreads();

            // continue if the coordinates of current thread is not supposed to calculate a value
            if(row >= n_local || col >= m_local ){
                __syncthreads();
                continue;
            }

            for (int vec_idx_in_ph = 0; vec_idx_in_ph < iteLen; vec_idx_in_ph++){
                total += ar_jv_ph[ty][vec_idx_in_ph]
                    * ar_img_k_ph[vec_idx_in_ph]
                    * ar_exp_ph[tx][vec_idx_in_ph];
            }

            __syncthreads();
        }

        if(row >= n_local || col >= m_local )
            return;
        ar_bnm[pos] = total;
    }""")
