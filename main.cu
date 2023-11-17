#include "shared.cuh"

int main(int argc, char **argv) {


    int world_rank, num_ranks;
    int proc_rows, proc_cols;
    bool prflag, pcflag;
    int n;
    bool nflag;
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-proc_rows", &proc_rows, &prflag));
    PetscCall(PetscCheck(prflag, "Must specify -proc_rows"));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-proc_cols", &proc_cols, &pcflag));
    PetscCall(PetscCheck(pcflag, "Must specify -proc_cols"));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, &nflag));
    PetscCall(PetscCheck(nflag, "Must specify -n"));


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
            a[i] = i;
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
    PetscCall(VecSetUp(v));

    printf("hello from rank %d\n", world_rank);

    PetscCall(VecView(v, PETSC_VIEWER_STDOUT_WORLD));

    Vec w;
    PetscCall(VecDuplicate(v, &w));
    PetscCall(VecZeroEntries(w));

    PetscCall(VecView(w, PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(VecAXPY(w, 1.0, v)); 


    double norm;
    PetscCall(VecNorm(v, NORM_2, &norm)); // replace with norm of w and doesnt hang
    PetscPrintf(PETSC_COMM_WORLD, "Norm: %f\n", norm);

    PetscCall(VecDestroy(&v));
    PetscCall(VecDestroy(&w));
    PetscCall(PetscLayoutDestroy(&layout));


    

    PetscCall(PetscFinalize());
    return 0;

}