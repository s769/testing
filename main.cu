#include "shared.cuh"



int main(int argc, char **argv) {

    Mat A;
    PetscViewer viewer;
    PetscInt N = 49000;
    PetscMPIInt rank;
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscInt loc_size = (rank == 0) ? N : 0;
    PetscCall(MatCreateDenseCUDA(PETSC_COMM_WORLD,loc_size,loc_size,PETSC_DETERMINE,PETSC_DETERMINE,NULL,&A));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "A.dat", FILE_MODE_WRITE, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
    PetscCall(MatView(A, viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    
    PetscCall(MatDestroy(&A));


    PetscCall(PetscFinalize());
    return 0;
}



