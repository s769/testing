#include "shared.cuh"

int main(int argc, char **argv) {


    int world_rank, num_ranks;
    int proc_rows, proc_cols;
    bool prflag, pcflag;
    int nt, nm;
    bool nflag;
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-proc_rows", &proc_rows, NULL));
    // PetscCall(PetscCheck(prflag, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must specify -proc_rows"));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-proc_cols", &proc_cols, NULL));
    // PetscCall(PetscCheck(pcflag,  PETSC_COMM_WORLD, PETSC_ERR_USER, "Must specify -proc_cols"));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-nt", &nt, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-nm", &nm, NULL));

    // PetscCall(PetscCheck(nflag,  PETSC_COMM_WORLD, PETSC_ERR_USER, "Must specify -n"));


    PetscCall(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    PetscCall(MPI_Comm_size(MPI_COMM_WORLD, &num_ranks));

    int row_rank = world_rank / proc_cols;
    int col_rank = world_rank % proc_cols;
    double *a, *d_a;
    int nm_local = (col_rank < nm % proc_cols) ? nm / proc_cols + 1 : nm / proc_cols;

    int before_me = (col_rank < nm % proc_cols) ? (nm/proc_cols + 1) * col_rank : (nm/proc_cols + 1) * nm % proc_cols + (nm/proc_cols) * (col_rank - nm % proc_cols);

    Vec v;
    if (row_rank == 0)
    {
        a = (double *) malloc(nm_local * nt * sizeof(double));
        for (int i = 0; i < nm_local; i++) {
            for (int j = 0; j < nt; j++) {
                a[i*nt + j] = before_me*nt + i*nt + j;
            }
        }
        cudaMalloc(&d_a, nm_local * nt * sizeof(double));
        cudaMemcpy(d_a, a, nm_local * nt * sizeof(double), cudaMemcpyHostToDevice);
        free(a);

    }

    PetscCall(VecCreate(PETSC_COMM_WORLD, &v));
    PetscLayout layout;
    PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layout));
    PetscCall(PetscLayoutSetBlockSize(layout, 1));
    if (row_rank == 0)
        PetscCall(PetscLayoutSetLocalSize(layout, nm_local * nt));
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

    VecScatter scatter;
    Vec v2;
    IS is;
    
    PetscLayout layout2;
    PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layout2));
    PetscCall(PetscLayoutSetBlockSize(layout2, 1));
    int nt_local = (world_rank < nt % num_ranks) ? nt / num_ranks + 1 : nt / num_ranks;
    int before_me2 = (world_rank < nt % num_ranks) ? (nt/num_ranks + 1) * world_rank : (nt/num_ranks + 1) * (nt % num_ranks) + (nt/num_ranks) * (world_rank - nt % num_ranks);

    printf("world_rank: %d, nt_local: %d, before_me2: %d\n", world_rank, nt_local, before_me2);
    if (col_rank == 0)
        PetscCall(PetscLayoutSetLocalSize(layout2, nt_local * nm));
    else
        PetscCall(PetscLayoutSetLocalSize(layout2, 0));
    
    PetscCall(PetscLayoutSetUp(layout2));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &v2));
    PetscCall(VecSetLayout(v2, layout2));
    PetscCall(VecSetType(v2, VECCUDA));
    PetscCall(VecSetUp(v2));

    int* idx = new int[nt_local * nm];

    for (int i = 0; i < nt_local; i++)
    {
        for (int j = 0; j < nm; j++)
        {
            idx[i*nm + j] = before_me2 + i + j*nt;
            printf("world_rank: %d, idx: %d\n", world_rank, idx[i*nm + j]);
        }
    }



    delete[] idx;

    PetscCall(VecDestroy(&v));
    PetscCall(PetscLayoutDestroy(&layout));





    

    PetscCall(PetscFinalize());
    return 0;

}