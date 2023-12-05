#include "shared.cuh"

int main(int argc, char **argv) {


    int world_rank, num_ranks;
    int proc_rows, proc_cols;
    bool prflag, pcflag;
    int n;
    bool nflag;
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-proc_rows", &proc_rows, NULL));
    // PetscCall(PetscCheck(prflag, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must specify -proc_rows"));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-proc_cols", &proc_cols, NULL));
    // PetscCall(PetscCheck(pcflag,  PETSC_COMM_WORLD, PETSC_ERR_USER, "Must specify -proc_cols"));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
    // PetscCall(PetscCheck(nflag,  PETSC_COMM_WORLD, PETSC_ERR_USER, "Must specify -n"));


    PetscCall(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    PetscCall(MPI_Comm_size(MPI_COMM_WORLD, &num_ranks));

    int row_rank = world_rank / proc_cols;
    int col_rank = world_rank % proc_cols;
    double *a, *d_a;

    Vec v;
    if (row_rank == 0)
    {
        a = (double *) malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) {
            a[i] = world_rank*n + i + 1;
        }
        cudaMalloc(&d_a, n * sizeof(double));
        cudaMemcpy(d_a, a, n * sizeof(double), cudaMemcpyHostToDevice);
        free(a);

    }

    PetscCall(VecCreate(PETSC_COMM_WORLD, &v));
    PetscLayout layout;
    PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layout));
    PetscCall(PetscLayoutSetBlockSize(layout, 1));
    if (row_rank == 0)
        PetscCall(PetscLayoutSetLocalSize(layout, n/proc_cols));
    else
        PetscCall(PetscLayoutSetLocalSize(layout, 0));
    PetscCall(PetscLayoutSetUp(layout));
    PetscCall(VecSetLayout(v, layout));
    PetscCall(VecSetType(v, VECCUDA));


    if (row_rank == 0)
    {
        PetscCall(VecCUDAReplaceArray(v, d_a));
    }
    else
    {
        PetscCall(VecCUDAReplaceArray(v, NULL));
    }
    PetscCall(VecSetUp(v));


    PetscCall(VecView(v, PETSC_VIEWER_STDOUT_WORLD));





    PetscCall(VecDestroy(&v));
    PetscCall(PetscLayoutDestroy(&layout));





    

    PetscCall(PetscFinalize());
    return 0;

}