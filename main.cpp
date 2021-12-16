#include <chrono>
#include <cmath>
#include <mpi.h>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>

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
    double ns;
    double ns_per_iter;
};

std::vector<int> random_vec(std::uint64_t n, std::uniform_int_distribution<int>& uid, std::mt19937& rng, std::uint64_t myid, size_t numprocs)
{
    std::vector<int> res;
    res.resize(n);
    for(std::size_t i = 0; i < n; i += 1)
    {
        res[i] = uid(rng);
    }
    return res;

}

int main(int argc, char **argv)
{
  int myid, numprocs;
  int n = 16*1000*1000ll*100; //vector len

  std::random_device rd;
  static std::mt19937 rng(rd());
  std::uniform_int_distribution<int> uid(-5,5);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;

  auto time_start = std::chrono::high_resolution_clock::now();
  auto time_end = time_start;

  auto time_now = std::chrono::system_clock::to_time_t(time_start);
  MPI_Get_processor_name(processor_name, &name_len);
  std::cout << "I'm #" << myid << " of " << numprocs <<" on " << processor_name << " at " <<  std::ctime(&time_now) << '\n';

  std::vector<int> v1, v2, local_v1, local_v2;
  int local_n = 0;
  int local_dot = 0, dot = 0;

  //generation start
  v1.resize(n);
  v2.resize(n);
  if(myid==0)
  {
      local_n = v1.size() / (unsigned)numprocs;
  }
  MPI_Bcast(&local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  local_v1.resize(local_n);
  local_v2.resize(local_n);
  for (int i = 0; i < local_n; i++)
  {
   local_v1[i]=uid(rng);
   local_v2[i]=uid(rng);
  }
  MPI_Gather(local_v1.data(), local_n, MPI_INT, v1.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(local_v2.data(), local_n, MPI_INT, v2.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
  //generation end

  unsigned long ticks_start, ticks_end;
  if(myid==0)
  {
      asm volatile("# mesurement start");
      time_start = std::chrono::high_resolution_clock::now();
      ticks_start = ticks();
      local_n = v1.size() / numprocs;
  }
  MPI_Bcast(&local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  local_v1.resize(local_n);
  local_v2.resize(local_n);
  MPI_Scatter(v1.data(), local_n, MPI_INT, local_v1.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatter(v2.data(), local_n, MPI_INT, local_v2.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

  #pragma omp parallel
  for (int i = 0; i < local_n; i++)
  {
   local_v1[i] *= local_v2[i];
  }
  #pragma omp parallel
  for (auto i : local_v1)
  {
      local_dot += i;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&local_dot, &dot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if(myid==0)
  {
      time_end = std::chrono::high_resolution_clock::now();
      ticks_end = ticks();
      asm volatile("# mesurement end");
      int iterations = 1;
      Timings result {
              .iterations = iterations,
              .ns =  static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start).count()),
              .ns_per_iter = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start).count())/iterations
                          };
      std::cout << "iter,ms,ms/iter" << '\n';
      std::cout << result.iterations << ',' << result.ns/1000000 << ',' << result.ns_per_iter/1000000 << '\n';
      std::cout << "dot is " << dot << '\n';
  }

  MPI_Finalize();
  return 0;
}


