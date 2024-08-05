#include "shared.cuh"



int main(int argc, char **argv) {

    Mat A;
    PetscViewer viewer;
    PetscInt N = 49000;
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
    PetscCall(MatCreateDenseCUDA(PETSC_DECIDE,PETSC_DECIDE,N,N,NULL,&A));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "A.dat", FILE_MODE_WRITE, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
    PetscCall(MatView(A, viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    
    PetscCall(MatDestroy(&A));


    PetscCall(PetscFinalize());
    return 0;
}



