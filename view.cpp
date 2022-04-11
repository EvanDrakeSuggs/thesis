/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <vector>

#include <Kokkos_Core.hpp>
#include <mpi.h>

int main(int argc, char* argv[]) {	
	  MPI_Init(&argc, &argv);
  Kokkos::initialize( argc, argv );
  {

	  //int *rank = (int*) malloc(sizeof(int));
	  int rank = 0;
	  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	  Kokkos::View<int[8]> check ("check");//[4][3]
	  Kokkos::parallel_for(check.extent(0), KOKKOS_LAMBDA(int i){
			  check(i) = i;
			  std::cout << i << "th entry: " << check(i) << std::endl;
		  });
	// template <typename DataType>
	// get layout algo, should work regardless of type,etc
	int size = check.size();
	int len = 0;
	int ex = -1;
	std::vector<int> v;
	while(size != 1){
		ex = check.extent(len);
		size = size/ex;
		v.push_back(ex);
		len++;
	}
	for(int i: v)
		std::cout << i << std::endl;
	//int strides[3];
	//check.stride(strides);
	//std::cout << check.stride_0() << std::endl;
	using traits = Kokkos::ViewTraits<int>;
	constexpr traits::array_layout layout;
	//using lay = traits::array_layout;
	// int num = 0;
	
	Kokkos::parallel_for(check.extent(0), KOKKOS_LAMBDA(int i)
	   {
		   int num = 0;
		   if(rank == 0){
			   num = i;
			   MPI_Send(&num, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
	   }
		   else if(rank == 1){
		MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		std::cout << i <<"th index, num: " << num << std::endl;	   
		   };
	   });
	
	/* func proto MPI_View_Send(view, MPI_INT, ...)
	   parallel_for(...,KOKKOS_LAMBDA(T i){
	   Toolbox::generic_p2p_blocking(....)
	   })*/

  }
  Kokkos::finalize();
  	MPI_Finalize();
  return 0;//(sum == seqSum) ? 0 : -1;
}
