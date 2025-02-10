#include "lodepng.h"
#include "matrix.h"
#include "math_utils.h"

#include <sstream>
#include <math.h>

#define N_POINTS 512
#define DOMAIN_SIZE 1.0f
#define N_ITERATIONS 10000
#define STAB_SAFETY_FACTOR 2
#define KINEMATIC_VISCOSITY 0.05f
#define DENSITY 1.0f
#define N_PRESSURE_POISSON_ITERATIONS 400

// Velocity Boundary Conditions
__global__ void set_velocity_bcs_vertical_kernel(
    float* u,
    float* v, 
    int cols
) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col >= cols) return;

    // Left Wall - Dirichlet
    int hole_bot = cols / 2 - int(cols/10);
    int hole_top = cols / 2 + int(cols/10);
    if ( (col <= hole_bot) || (col >= hole_top) )
        u[0 * cols + col] = 0.0f;

    v[0 * cols + col] = 0.0f;

    // Right Wall - Homogeneous Dirichlet
    u[(N_POINTS - 1) * cols + col] = 0.0f;
    v[(N_POINTS - 1) * cols + col] = 0.0f;
}

__global__ void set_velocity_bcs_horizontal_kernel(
    float* u,
    float* v,
    int rows,
    int cols
) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows) return;

    // Bottom Wall
    u[row * cols + (0)] = 0.0f;
    v[row * cols + (0)] = 0.0f;

    // Top Wall
    u[row * cols + (N_POINTS - 1)] = 0.0f;
    v[row * cols + (N_POINTS - 1)] = 0.0f;
}

void set_velocity_bcs(Matrix& u, Matrix& v) {
    int threads = (N_POINTS < tx) ? N_POINTS : tx;
    int blocks  = N_POINTS / threads + 1;

    set_velocity_bcs_vertical_kernel<<<blocks, threads>>>(u.data, v.data, u.cols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    set_velocity_bcs_horizontal_kernel<<<blocks, threads>>>(u.data, v.data, u.rows, u.cols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

// Pressure Boundary Conditions
__global__ void set_pressure_bcs_vertical_kernel(
    float* p, 
    int cols
) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col >= cols) return;

    // Left Wall - Dirichlet
    int hole_bot = cols / 2 - int(cols/10);
    int hole_top = cols / 2 + int(cols/10);
    if ( (col > hole_bot) && (col < hole_top) )
        p[0 * cols + col] = 200.0f;
    else
        p[0 * cols + col] = 0.0f; // p[1 * cols + col];

    // Right Wall - Homogeneous Dirichlet
    p[(N_POINTS - 1) * cols + col] = 0.0f;
    // p[(N_POINTS - 1) * cols + col] = p[(N_POINTS - 2) * cols + col];
}

__global__ void set_pressure_bcs_horizontal_kernel(
    float* p,
    int rows,
    int cols
) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows) return;

    // Bottom Wall
    p[row * cols + (0)] = 0.0f; // p[row * cols + (1)];

    // Top Wall
    p[row * cols + (N_POINTS - 1)] = 0.0f; // p[row * cols + (N_POINTS - 2)];
}

void set_pressure_bcs(Matrix& p) {
    int threads = (N_POINTS < tx) ? N_POINTS : tx;
    int blocks  = N_POINTS / threads + 1;

    set_pressure_bcs_vertical_kernel<<<blocks, threads>>>(p.data, p.cols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    set_pressure_bcs_horizontal_kernel<<<blocks, threads>>>(p.data, p.rows, p.cols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

__global__ void add_smoke_source_kernel(float* smoke_field, int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    int smoke_source_bot = cols / 2 - int(N_POINTS / 10);
    int smoke_source_top = cols / 2 + int(N_POINTS / 10);

    if (
        (col > smoke_source_bot)
        && (col < smoke_source_top)
    ) {
        int idx = 0 * cols + col;
        smoke_field[idx] = DENSITY;
    }
}
void add_smoke_source(Matrix& smoke_field) {
    int threads = (N_POINTS < tx) ? N_POINTS : tx;
    int blocks  = N_POINTS / threads + 1;

    add_smoke_source_kernel<<<blocks, threads>>>(smoke_field.data, smoke_field.cols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

// Helper function to write PNG
void output_png(
    const Matrix& d_flow_variable,
    std::string   outfile_name,
    const float   vmax,
    const float   vmin,
    const float   color_start[3],
    const float   color_end[3]
) {
    // Get a copy of the flow variable on the CPU
    float *flow_variable = new float[d_flow_variable.rows * d_flow_variable.cols]();
    CHECK_CUDA(cudaMemcpy(flow_variable, d_flow_variable.data, d_flow_variable.n_bytes, cudaMemcpyDeviceToHost));

    // Initialize image buffer
    std::vector<unsigned char> image;
    image.resize(d_flow_variable.rows * d_flow_variable.cols * 4);

    for (int i = 0; i < d_flow_variable.rows; i++) {
        for (int j = 0; j < d_flow_variable.cols; j++) {
            // Map flow_variable to 0..1
            float t = (flow_variable[i * d_flow_variable.cols + j] - vmin) / (vmax - vmin);
            t = std::fminf( std::fmaxf( 0.0f, t ), 1.0f);

            // Lerp to get color
            float rf = color_start[0] * (1.0f - t) + color_end[0] * t;
            float gf = color_start[1] * (1.0f - t) + color_end[1] * t;
            float bf = color_start[2] * (1.0f - t) + color_end[2] * t;

            // Convert to ints
            int r = int(rf * 255.99);
            int g = int(gf * 255.99);
            int b = int(bf * 255.99);

            // Write into image buffer
            int pixel_index = (N_POINTS - j - 1) * d_flow_variable.rows + i; // Flip y-direction and transpose
            image[4 * pixel_index + 0] = r;
            image[4 * pixel_index + 1] = g;
            image[4 * pixel_index + 2] = b;
            image[4 * pixel_index + 3] = 255;
        }
    }

    unsigned error = lodepng::encode(outfile_name, image, d_flow_variable.cols, d_flow_variable.rows);
    if(error) 
        std::cerr << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}

int main () {

    // Image output
    // float color_start[] = {0.533f, 0.800f, 0.933f};
    // float color_end[]   = {0.933f, 0.533f, 0.600f};
    float color_start[] = {1.000f, 1.000f, 1.000f};
    float color_end[]   = {0.933f, 0.233f, 0.300f};
    
    // Grid spacing
    float element_length = (DOMAIN_SIZE - 0.0f) / (N_POINTS - 1);

    // Max time step
    float max_time_step_len_diss = 0.5f * element_length * element_length / KINEMATIC_VISCOSITY;
    float TIME_STEP_LENGTH  = max_time_step_len_diss / STAB_SAFETY_FACTOR;

    std::cout << "Using timestep of size " << TIME_STEP_LENGTH << "." << std::endl;

    // Initialize pressure/velocity
    Matrix u_prev(N_POINTS, N_POINTS);
    Matrix v_prev(N_POINTS, N_POINTS);
    Matrix p_prev(N_POINTS, N_POINTS);

    Matrix p_next(N_POINTS, N_POINTS);

    // Initialize smoke field (for visualization)
    Matrix smoke_field(N_POINTS, N_POINTS);
    add_smoke_source(smoke_field);

    // Initialize derivative terms
    Matrix d_u_prev__dx(N_POINTS, N_POINTS);
    Matrix d_u_prev__dy(N_POINTS, N_POINTS);
    Matrix d_v_prev__dx(N_POINTS, N_POINTS);
    Matrix d_v_prev__dy(N_POINTS, N_POINTS);
    Matrix laplace__u_prev(N_POINTS, N_POINTS);
    Matrix laplace__v_prev(N_POINTS, N_POINTS);

    Matrix d_u_tent__dx(N_POINTS, N_POINTS);
    Matrix d_u_tent__dy(N_POINTS, N_POINTS);
    Matrix d_v_tent__dx(N_POINTS, N_POINTS);
    Matrix d_v_tent__dy(N_POINTS, N_POINTS);

    Matrix d_smoke__dx(N_POINTS, N_POINTS);
    Matrix d_smoke__dy(N_POINTS, N_POINTS);

    // Initialize tentative velocity matrices
    Matrix u_tent(N_POINTS, N_POINTS);
    Matrix v_tent(N_POINTS, N_POINTS);

    // Initialize pressure corrector rhs matrix
    Matrix pcorr_rhs(N_POINTS, N_POINTS);

    // Initialize pressure gradient terms (for pressure corrector)
    Matrix d_p_next__dx(N_POINTS, N_POINTS);
    Matrix d_p_next__dy(N_POINTS, N_POINTS);

    // Iterate
    for (int iteration = 0; iteration < N_ITERATIONS; iteration++) {        
        // Calculate derivatives
        interior_central_difference_x_uniform(u_prev, element_length, d_u_prev__dx);
        interior_central_difference_y_uniform(u_prev, element_length, d_u_prev__dy);
        interior_central_difference_x_uniform(v_prev, element_length, d_v_prev__dx);
        interior_central_difference_y_uniform(v_prev, element_length, d_v_prev__dy);
        interior_laplace_5point_stencil_uniform(u_prev, element_length, laplace__u_prev);
        interior_laplace_5point_stencil_uniform(v_prev, element_length, laplace__v_prev);

        // Perform the estimator step -- solve momentum discretization without pressure gradient
        u_tent = u_prev + TIME_STEP_LENGTH * (
            KINEMATIC_VISCOSITY * laplace__u_prev
            - (u_prev * d_u_prev__dx + v_prev * d_u_prev__dy)
        );
        v_tent = v_prev + TIME_STEP_LENGTH * (
            KINEMATIC_VISCOSITY * laplace__v_prev
            - (u_prev * d_v_prev__dx + v_prev * d_v_prev__dy)
        );

        // Enforce Velocity Boundary Conditions
        set_velocity_bcs(u_tent, v_tent);

        // Pressure Corrector Step
        interior_central_difference_x_uniform(u_tent, element_length, d_u_tent__dx);
        interior_central_difference_y_uniform(v_tent, element_length, d_v_tent__dy);

        pcorr_rhs = DENSITY / TIME_STEP_LENGTH * ( d_u_tent__dx + d_v_tent__dy );

        // Solve pressure Poisson problem via Jacobi iterations
        for (int jacobi_iter = 0; jacobi_iter < N_PRESSURE_POISSON_ITERATIONS; jacobi_iter++) {
            p_next.zero();

            pressure_poisson_jacobi_iteration(p_prev, pcorr_rhs, element_length, p_next);

            // Pressure BCs
            set_pressure_bcs(p_next);

            p_prev = p_next;
        }
        
        // Correct Velocity
        interior_central_difference_x_uniform(p_next, element_length, d_p_next__dx);
        interior_central_difference_y_uniform(p_next, element_length, d_p_next__dy);

        // Correct the interior
        u_prev = u_tent
            - TIME_STEP_LENGTH / DENSITY
            * d_p_next__dx;
        v_prev = v_tent
            - TIME_STEP_LENGTH / DENSITY
            * d_p_next__dy;

        // Enforce the boundary conditions
        set_velocity_bcs(u_prev, v_prev);

        // Update the smoke field
        interior_central_difference_x_uniform(smoke_field, element_length, d_smoke__dx);
        interior_central_difference_y_uniform(smoke_field, element_length, d_smoke__dy);
        smoke_field = smoke_field - TIME_STEP_LENGTH * (
            u_prev * d_smoke__dx + v_prev * d_smoke__dy
            + smoke_field * ( d_u_prev__dx + d_v_prev__dy )
        );
        add_smoke_source(smoke_field);
        
        if (iteration % 100 == 0) {
            std::cout << "Time Step " << iteration + 1 << "/" << N_ITERATIONS << ": " << "\n";
            std::cout << "> Max p " << p_next.max() << ", Max u " << u_prev.max() << ", Max v " << v_prev.max() << "\n";
            std::cout << "> Min p " << p_next.min() << ", Min u " << u_prev.min() << ", Min v " << v_prev.min() << "\n\n";
        }

        if (iteration % 50 == 0) {
            // WRITE IMAGES TO DISK
            std::stringstream p_outfile;
            p_outfile << "pres_anim/" << iteration << ".png";
            output_png(
                p_prev,
                p_outfile.str(),
                100.0f,
                0.0f,
                color_start,
                color_end
            );

            std::stringstream u_outfile;
            u_outfile << "u_anim/" << iteration << ".png";
            output_png(
                u_prev,
                u_outfile.str(),
                3,
                -3,
                color_start,
                color_end
            );

            std::stringstream v_outfile;
            v_outfile << "v_anim/" << iteration << ".png";
            output_png(
                v_prev,
                v_outfile.str(),
                2,
                -2,
                color_start,
                color_end
            );
            
            std::stringstream smoke_outfile;
            smoke_outfile << "smoke_anim/" << iteration << ".png";
            output_png(
                smoke_field,
                smoke_outfile.str(),
                DENSITY,
                0.0f,
                color_start,
                color_end
            );
        }

    }

    return 0;
}