#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>

inline void CUDA_CHK(const cudaError_t err){
  if(err != cudaSuccess){
    std::cout << "cuda call failed" << std::endl;
    std::exit(-1);
  }
}
inline void MPI_CHK(const int err){
  if(err != MPI_SUCCESS){
    std::cout << "mpi call failed" << std::endl;
    std::exit(-1);
  }
}

int main(int argc, char **argv){

  int count, size, rank;

  MPI_CHK(MPI_Init(&argc, &argv));
  MPI_CHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  
  for(int rankx=0 ; rankx<size ; rankx++){
    if (rankx == rank){
      std::cout << "Rank: " << rank << std::endl;

      CUDA_CHK(cudaGetDeviceCount(&count));
      std::cout << "\tNum devices: " << count << std::endl;
      
      for(int dx=0 ; dx<count ; dx++){
        cudaDeviceProp prop;
        CUDA_CHK(cudaGetDeviceProperties(&prop, dx));
        char pciBusId[1024];
        CUDA_CHK(cudaDeviceGetPCIBusId(pciBusId, 1024, dx)); 
        std::cout << "\tDevice: " << dx << std::endl;
        std::cout << "\t\tName: " << prop.name << std::endl;
        std::cout << "\t\tBus id: " << prop.pciBusID << std::endl;
        std::cout << "\t\tBus id: " << pciBusId << std::endl;
        std::cout << "\t\tDevice id: " << prop.pciDeviceID << std::endl;
        std::cout << "\t\tDomain id: " << prop.pciDomainID << std::endl;
      }
      std::cout << std::flush;
    }
    MPI_Barrier(MPI_COMM_WORLD);

  }

  MPI_CHK(MPI_Finalize());
  return 0;
}


