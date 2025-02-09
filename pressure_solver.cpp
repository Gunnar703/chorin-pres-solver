#include "lodepng.h"
#include "matrix.h"
#include "vectors.h"
#include "math_utils.h"

#include <sstream>
#include <math.h>

#define N_POINTS 201
#define DOMAIN_SIZE 1.0f
#define N_ITERATIONS 5000
#define STAB_SAFETY_FACTOR 2
#define KINEMATIC_VISCOSITY 0.05f
#define DENSITY 1.0f
#define N_PRESSURE_POISSON_ITERATIONS 50

// Set boundary conditions
void set_velocity_bcs(Matrix& u, Matrix& v) {
    for (int j = 0; j < u.cols; j++) {
        // Left Wall  -- Homogeneous Dirichlet (except for jet)
        if (j <= u.cols/2 - int(N_POINTS/10))
            u(0, j) = 0.0f;
        
        if (j >= u.cols/2 + int(N_POINTS/10))
            u(0, j) = 0.0f;

        u(N_POINTS - 1, j) = 0.0f; // Right Wall - Homogeneous
    
        v(0,            j) = 0.0f; // Left Wall
        v(N_POINTS - 1, j) = 0.0f; // Right Wall
    }

    for (int i = 0; i < u.rows; i++) {
        u(i,            0) = 0.0f;  // Bottom Wall - Homogeneous
        u(i, N_POINTS - 1) = 0.0f;  // Top Wall
    
        v(i,            0) = 0.0f;  // Bottom Wall - Homogeneous
        v(i, N_POINTS - 1) = 0.0f;  // Top Wall    - Homogeneous
    }
}

void set_pressure_bcs(Matrix& p) {
    for (int j = 0; j < p.cols; j++) {
        p(N_POINTS - 1, j) = p(N_POINTS - 2, j); // Right Wall -- Homogeneous Neumann
        
        // Left Wall  -- Dirichlet (except for jet)
        if ((j > p.cols/2 - int(N_POINTS/10)) && (j < p.cols/2 + int(N_POINTS/10)))
            p(0, j) = 100.0f;
        else
            p(0, j) = 0.0f;
    }

    for (int i = 0; i < p.rows; i++) {
        p(i,            0) = p(i, 1); 
        p(i, N_POINTS - 1) = p(i, N_POINTS - 2);
    }
}

void add_smoke_source(Matrix& smoke_field) {
    int smoke_source_bot = smoke_field.cols / 2 - int(N_POINTS / 10);
    int smoke_source_top = smoke_field.cols / 2 + int(N_POINTS / 10);

    for (int j = smoke_source_bot; j <= smoke_source_top; j++)
        smoke_field(0, j) = 1;
}

// Helper function to write PNG
void output_png(
    const Matrix& flow_variable,
    std::string   outfile_name,
    const float   vmax,
    const float   vmin,
    const Vector  color_start,
    const Vector  color_end
) {
    // Initialize image buffer
    std::vector<unsigned char> image;
    image.resize(flow_variable.rows * flow_variable.cols * 4);

    for (int i = 0; i < flow_variable.rows; i++) {
        for (int j = 0; j < flow_variable.cols; j++) {
            // Map flow_variable to 0..1
            float t = (flow_variable(i, j) - vmin) / (vmax - vmin);
            t = std::fminf( std::fmaxf( 0.0f, t ), 1.0f);

            // Lerp to get color
            Vector rgb = color_start * (1.0f - t) + color_end * t;

            // Convert to ints
            int r = int(rgb(0) * 255.99);
            int g = int(rgb(1) * 255.99);
            int b = int(rgb(2) * 255.99);

            // Write into image buffer
            int pixel_index = (N_POINTS - j - 1) * flow_variable.rows + i; // Flip y-direction and transpose
            image[4 * pixel_index + 0] = r;
            image[4 * pixel_index + 1] = g;
            image[4 * pixel_index + 2] = b;
            image[4 * pixel_index + 3] = 255;
        }
    }

    unsigned error = lodepng::encode(outfile_name, image, flow_variable.cols, flow_variable.rows);
    if(error) 
        std::cerr << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}

int main () {

    // Image output
    Vector color_start(3), color_end(3);
    color_start(0) = 0.533f;
    color_start(1) = 0.800f;
    color_start(2) = 0.933f;
    color_end(0)   = 0.933f;
    color_end(1)   = 0.533f;
    color_end(2)   = 0.600f;

    // X and Y coordinate vectors
    Vector x(N_POINTS), y(N_POINTS);
    x.linspace(0.0f, DOMAIN_SIZE);
    y.linspace(0.0f, DOMAIN_SIZE);
    
    // Grid spacing
    float element_length = x(1) - x(0);

    // Max time step
    float max_time_step_len_diss = 0.5f * element_length * element_length / KINEMATIC_VISCOSITY;
    float TIME_STEP_LENGTH  = max_time_step_len_diss / STAB_SAFETY_FACTOR;

    std::cout << "Using timstep of size " << TIME_STEP_LENGTH << "." << std::endl;

    // X and Y coordinate grids
    Matrix X(N_POINTS, N_POINTS), Y(N_POINTS, N_POINTS);
    meshgrid(x, y, X, Y);

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

    // Store all iterations
    // std::vector<Matrix> p_history;
    // std::vector<Matrix> u_history;
    // std::vector<Matrix> v_history;
    // std::vector<Matrix> smoke_history;

    // float p_min, p_max, u_min, u_max, v_min, v_max;

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

            for (int i = 1; i < p_next.rows - 1; i++) {
                for (int j = 1; j < p_next.cols - 1; j++) {
                    float pij = p_prev(i, j - 1) + p_prev(i, j + 1)
                              + p_prev(i - 1, j) + p_prev(i + 1, j);
                    float cor = element_length * element_length * pcorr_rhs(i, j);
                    p_next(i, j) = (pij - cor) / 4;
                }
            }

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
        // add_smoke_source(smoke_field);  // Enforce smoke BCs 

        // Advance
        p_prev = p_next;

        // Store
        // p_history.push_back(p_prev);
        // u_history.push_back(u_prev);
        // v_history.push_back(v_prev);
        // smoke_history.push_back(smoke_field);

        // Calculate global mins/maxes
        // if (iteration == 0) {
        //     p_min = p_max = p_next(0, 0);
        //     u_min = u_max = u_prev(0, 0);
        //     v_min = v_max = v_prev(0, 0);
        // } else {
        //     p_min = std::fminf( p_min, p_next.min() );
        //     u_min = std::fminf( u_min, u_prev.min() );
        //     v_min = std::fminf( v_min, v_prev.min() );
            
        //     p_max = std::fmaxf( p_max, p_next.max() );
        //     u_max = std::fmaxf( u_max, u_prev.max() );
        //     v_max = std::fmaxf( v_max, v_prev.max() );
        // }
        
        if (iteration % 100 == 0) {
            std::cout << "Time Step " << iteration + 1 << "/" << N_ITERATIONS << ": " << "\n";
            std::cout << "> Max p " << p_next.max() << ", Max u " << u_prev.max() << ", Max v " << v_prev.max() << "\n";
            std::cout << "> Min p " << p_next.min() << ", Min u " << u_prev.min() << ", Min v " << v_prev.min() << "\n\n";
        }

        if (iteration % 10 == 0) {
            // WRITE IMAGES TO DISK
            std::stringstream p_outfile;
            p_outfile << "pres_anim/" << iteration << ".png";
            output_png(
                p_prev,
                p_outfile.str(),
                12.0f,
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
                3,
                -3,
                color_start,
                color_end
            );
            
            std::stringstream smoke_outfile;
            smoke_outfile << "smoke_anim/" << iteration << ".png";
            output_png(
                smoke_field,
                smoke_outfile.str(),
                1.0f,
                0.0f,
                color_start,
                color_end
            );
        }

    }

    return 0;
}