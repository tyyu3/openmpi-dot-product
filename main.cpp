#include <chrono>
#include <cmath>
#include <mpi.h>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>


static constexpr std::size_t iterations = 1;

template <typename T>
inline __attribute__((always_inline)) void do_not_optimize(T& value)
{
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

inline __attribute__((always_inline)) std::uint64_t ticks()
{
    std::uint64_t tsc;
    asm volatile("mfence; "         // memory barrier
                 "rdtsc; "          // read of tsc
                 "shl $32,%%rdx; "  // shift higher 32 bits stored in rdx up
                 "or %%rdx,%%rax"   // and or onto rax
                 : "=a"(tsc)        // output to tsc
                 :
                 : "%rcx", "%rdx", "memory");
    return tsc;
}

struct Timings
{
    std::size_t iterations;
    double ticks_per_iter;
    double ns_per_iter;
};


std::vector<int> random_vec(std::uint64_t n, std::uniform_int_distribution<int>& uid, std::mt19937& rng, std::uint64_t myid, size_t numprocs)
{
    std::vector<int> res;
    res.resize(n);
    for(std::size_t i = myid; i < n; i +=numprocs)
    {
        res[i] = uid(rng);
    }
    return res;
}

std::int64_t dot_product(const std::vector<int>& v1, const std::vector<int>&v2, std::uint64_t myid, size_t numprocs)
{
    std::int64_t res = 0, finalres = 0;
    for(std::int64_t i = myid; i < v1.size(); i +=numprocs)
    {

        //std::cout << res <<"+="<<v1[i]<<"*"<<v2[i] << std::endl;
        res += v1[i]*v2[i];
    }
    //std::cout << "local res " << res << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&res, &finalres, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    return finalres;
}


int main(int argc, char **argv)
{
  int done = 0, myid, numprocs;
  std::uint64_t n = 100; //vector len

  static std::mt19937 rng{1};
  std::uniform_int_distribution<int> uid(-5,5);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  std::cout << "I'm #" << myid << " of " << numprocs <<" on " << processor_name << '\n';

  auto v1 = random_vec(n, uid, rng, myid, numprocs);
  auto v2 = random_vec(n, uid, rng, myid, numprocs);

  /*for(auto i : v1)
  {
      std::cout<< i << std::endl;
  }
  std::cout <<"\n";
  for(auto i : v2)
  {
      std::cout<< i << std::endl;
  }*/

  auto time_start = std::chrono::high_resolution_clock::now();;
  auto time_end = std::chrono::high_resolution_clock::now();
  unsigned long ticks_start, ticks_end;
  //MPI_Bcast(&s, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::int64_t dot;
  {
      if(myid==0)
      {

          asm volatile("# mesurement start");
          time_start = std::chrono::high_resolution_clock::now();
          ticks_start = ticks();
      }
      //--------------------
      //for (std::size_t i = 0; i < iterations; ++i)
      //{
          dot = dot_product(v1, v2, myid, numprocs);
          do_not_optimize(dot);
      //}
      //--------------------
      if(myid==0)
      {
          time_end = std::chrono::high_resolution_clock::now();
          ticks_end = ticks();
          asm volatile("# mesurement end");

          Timings result {
                  .iterations = iterations,
                  .ticks_per_iter = static_cast<double>(ticks_end - ticks_start)/iterations,
                  .ns_per_iter = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start).count())/iterations
                              };
          std::cout << "iter,ns/iter,ticks/iter" << '\n';
          std::cout << result.iterations << ',' << result.ns_per_iter << ',' << result.ticks_per_iter << '\n';
          std::cout << "dot is " << dot << '\n';
      }
  }
  return 0;
}


