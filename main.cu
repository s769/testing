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

    int before_me = (col_rank < nm % proc_cols) ? (nm/proc_cols + 1) * col_rank : (nm/proc_cols + 1) * (nm % proc_cols) + (nm/proc_cols) * (col_rank - nm % proc_cols);

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
    // PetscLayout layout;
    // PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layout));
    // PetscCall(PetscLayoutSetBlockSize(layout, 1));
    // if (row_rank == 0)
    //     PetscCall(PetscLayoutSetLocalSize(layout, nm_local * nt));
    // else
    //     PetscCall(PetscLayoutSetLocalSize(layout, 0));

    int sz = (row_rank == 0) ? nm_local * nt : 0;
    PetscCall(VecSetSizes(v, sz, PETSC_DECIDE));
    // PetscCall(PetscLayoutSetUp(layout));
    // PetscCall(VecSetLayout(v, layout));
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
    
    // PetscLayout layout2;
    // PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layout2));
    // PetscCall(PetscLayoutSetBlockSize(layout2, 1));
    int nt_local = (world_rank < nt % num_ranks) ? nt / num_ranks + 1 : nt / num_ranks;
    int before_me2 = (world_rank < nt % num_ranks) ? (nt/num_ranks + 1) * world_rank : (nt/num_ranks + 1) * (nt % num_ranks) + (nt/num_ranks) * (world_rank - nt % num_ranks);


    // PetscCall(PetscLayoutSetLocalSize(layout2, nt_local * nm));


    
    // PetscCall(PetscLayoutSetUp(layout2));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &v2));
    // PetscCall(VecSetLayout(v2, layout2));
    // PetscCall(VecSetType(v2, VECCUDA));
    PetscCall(VecSetSizes(v2, nt_local * nm, PETSC_DECIDE));
    PetscCall(VecSetUp(v2));

    PetscCall(VecView(v2, PETSC_VIEWER_STDOUT_WORLD));
    int * indices;
    int rstart, rend;
    PetscCall(VecGetOwnershipRange(v2, &rstart, &rend));
    PetscCall(PetscMalloc1(rend - rstart, &indices));



    for (int i = 0; i < nt_local; i++)
    {
        for (int j = 0; j < nm; j++)
        {
            indices[i*nm + j] =  before_me2 + i + j*nt;
        }
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, rend - rstart, indices, PETSC_OWN_POINTER, &is));
// 



    // PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, nt_local * nm, indices, PETSC_USE_POINTER, &is));
    PetscCall(ISView(is, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecScatterCreate(v, is, v2, NULL, &scatter));
    PetscCall(VecScatterBegin(scatter, v, v2, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter, v, v2, INSERT_VALUES, SCATTER_FORWARD));



    PetscCall(VecView(v2, PETSC_VIEWER_STDOUT_WORLD));


    
    PetscCall(VecScatterDestroy(&scatter));



    PetscCall(VecDestroy(&v));
    PetscCall(VecDestroy(&v2));
    // PetscCall(PetscLayoutDestroy(&layout));
    // PetscCall(PetscLayoutDestroy(&layout2));
    PetscCall(ISDestroy(&is));
    // delete[] idx;
    





    

    PetscCall(PetscFinalize());
    return 0;

}

// int main(int argc, char **argv)
// {
//   PetscInt    i, j, rstart, rend, n, N, *indices;
//   PetscMPIInt size, rank;
//   IS          ix;
//   VecScatter  vscat;
//   Vec         x, y;

//   PetscFunctionBeginUser;
//   PetscCall(PetscInitialize(&argc, &argv, (char *)0, NULL));
//   PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
//   PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

//   PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
//   PetscCall(VecSetType(x, VECCUDA));
// //   PetscCall(VecSetFromOptions(x));
//   PetscCall(PetscObjectSetName((PetscObject)x, "Vec X"));
//   n = (rank < 3) ? 12 : 0;
//   PetscCall(VecSetSizes(x, n, PETSC_DECIDE));

//   PetscCall(VecGetOwnershipRange(x, &rstart, &rend));
//   for (i = rstart; i < rend; i++) PetscCall(VecSetValue(x, i, (PetscScalar)i, INSERT_VALUES));
//   PetscCall(VecAssemblyBegin(x));
//   PetscCall(VecAssemblyEnd(x));
//   PetscCall(VecGetSize(x, &N));
  

//   PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
//   PetscCall(VecSetType(y, VECCUDA));
// //   PetscCall(VecSetFromOptions(y));
//   PetscCall(PetscObjectSetName((PetscObject)y, "Vec Y"));
//   PetscCall(VecSetSizes(y, PETSC_DECIDE, N));

//   PetscCall(VecView(y, PETSC_VIEWER_STDOUT_WORLD));


//   PetscCall(VecGetOwnershipRange(y, &rstart, &rend));
//   PetscCall(PetscMalloc1(rend - rstart, &indices));
//   for (i = rstart, j = 0; i < rend; i++, j++) indices[j] = rank + size * j;

//   PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, rend - rstart, indices, PETSC_OWN_POINTER, &ix));
//   PetscCall(VecScatterCreate(x, ix, y, NULL, &vscat));

//   PetscCall(VecScatterBegin(vscat, x, y, INSERT_VALUES, SCATTER_FORWARD));
//   PetscCall(VecScatterEnd(vscat, x, y, INSERT_VALUES, SCATTER_FORWARD));

//   PetscCall(ISView(ix, PETSC_VIEWER_STDOUT_WORLD));
//   PetscCall(VecView(x, PETSC_VIEWER_STDERR_WORLD));
//   PetscCall(VecView(y, PETSC_VIEWER_STDERR_WORLD));

//   PetscCall(VecScatterDestroy(&vscat));
//   PetscCall(ISDestroy(&ix));
//   PetscCall(VecDestroy(&x));
//   PetscCall(VecDestroy(&y));
//   PetscCall(PetscFinalize());
//   return 0;
// }
