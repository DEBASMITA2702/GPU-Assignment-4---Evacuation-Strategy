#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <climits>
using namespace std;

// these constants are used to define the size of various arrays in the problem
#define MAX_DROPS 1000
#define MAX_PATH 1000
#define MAX_ATTEMPTS 10000
#define LOCK_FREE -1

// device function to perform atomicSub for long long datatype
__device__ long long atomicSubLL(long long *address, long long val)
{
    auto ptr = (unsigned long long *)address;
    unsigned long long old = *ptr, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(ptr, assumed, assumed - val);
    } while (assumed != old);
    return (long long)old;
}

// device function to perform atomicAdd for long long datatype
__device__ long long atomicAddLL(long long *address, long long val)
{
    auto ptr = (unsigned long long *)address;
    unsigned long long old = *ptr, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(ptr, assumed, assumed + val);
    } while (assumed != old);
    return (long long)old;
}

// kernel to initialize the distance and the parent arrays for the bellman ford execution
__global__ void init_kernel(int *dist_flat, int *prev_flat, long long *shelters, int num_shelters, int num_cities)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_shelters)
        return;
    int base = s * num_cities;
    int src = (int)shelters[s];
    for (int i = 0; i < num_cities; ++i)
    {
        dist_flat[base + i] = (i == src ? 0 : INT_MAX);
        prev_flat[base + i] = -1;
    }
}

// kernel to run the edge relaxation of all the edges during one iteration of the Bellman ford
__global__ void relax_edges(int *dist_flat, int *prev_flat, int *roads, int num_roads, int num_cities, long long *shelters, int num_shelters)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_shelters * num_roads;
    if (tid >= total)
        return;
    int s = tid / num_roads;
    int e = tid % num_roads;
    int base = s * num_cities;
    int u = roads[4 * e], v = roads[4 * e + 1], w = roads[4 * e + 2];
    int du = dist_flat[base + u];
    if (du != INT_MAX && du + w < dist_flat[base + v])
    {
        dist_flat[base + v] = du + w;
        prev_flat[base + v] = u;
    }
    int dv = dist_flat[base + v];
    if (dv != INT_MAX && dv + w < dist_flat[base + u])
    {
        dist_flat[base + u] = dv + w;
        prev_flat[base + u] = v;
    }
}

// function to find the best shelter for current origin
__device__ int findBestShelter(int origin, bool* visited, long long* shelter_caps, int* dist_flat, 
    int num_shelters, int num_cities, int max_dist_elderly, long long elderly, long long prime)
{
    int best_s = -1, best_d = INT_MAX;
    for (int s = 0; s < num_shelters; ++s)
    {
        if (visited[s] || shelter_caps[s] <= 0)
            continue;
        int d = dist_flat[s * num_cities + origin];
        if (d == INT_MAX || (elderly > 0 && d > max_dist_elderly && prime == 0))
            continue;
        if (d < best_d)
        {
            best_d = d;
            best_s = s;
        }
    }
    return best_s;
}

// device function to construct path from origin to shelter
__device__ int constructPath(int origin, int shelter_city, int* prev_flat, int s, int num_cities,
int* local_path, int max_path_size)
{
    int plen = 0, cur = origin;
    while (cur != -1 && plen < max_path_size)
    {
    local_path[plen++] = cur;
    if (cur == shelter_city)
        break;
    cur = prev_flat[s * num_cities + cur];
    }
    return plen;
}

// device function to merge the current path into the full path
__device__ int mergePaths(int* local_path, int plen, long long* full_path, int full_len)
{
    if (full_len == 0) {
        for (int k = 0; k < plen; ++k)
            full_path[full_len++] = local_path[k];
    } else {
        for (int k = 1; k < plen; ++k)
        full_path[full_len++] = local_path[k];
    }
    return full_len;
}

// function to handle dropping people at shelter
__device__ void processDrop(int tid, long long* shelter_caps, int best_s, int best_d, int max_dist_elderly,
long long* elderly_ptr, long long* prime_ptr, long long* num_drops, long long* drops, long long shelter_city, int origin)
{
    long long& elderly = *elderly_ptr;
    long long& prime = *prime_ptr;

    long long drop_e = (best_d <= max_dist_elderly ? elderly : 0);
    long long avail = atomicAddLL(&shelter_caps[best_s], 0);
    long long total = prime + drop_e, can = min(total, avail);

    if (can <= 0)
    return;

    atomicSubLL(&shelter_caps[best_s], can);
    drop_e = min(drop_e, can);
    elderly -= drop_e;
    long long drop_p = can - drop_e;
    prime -= drop_p;

    // add the drop to the result
    long long idx = atomicAddLL(&num_drops[tid], 1);
    drops[tid * MAX_DROPS * 3 + idx * 3 + 0] = shelter_city;
    drops[tid * MAX_DROPS * 3 + idx * 3 + 1] = drop_p;
    drops[tid * MAX_DROPS * 3 + idx * 3 + 2] = drop_e;

    // handle elderly people who cannot travel the distance
    if (elderly > 0 && best_d > max_dist_elderly)
    {
        long long idx2 = atomicAddLL(&num_drops[tid], 1);
        drops[tid * MAX_DROPS * 3 + idx2 * 3 + 0] = origin;
        drops[tid * MAX_DROPS * 3 + idx2 * 3 + 1] = 0;
        drops[tid * MAX_DROPS * 3 + idx2 * 3 + 2] = elderly;
        elderly = 0;
    }
}

// kernel to run the evacutation process (here each thread takes care of the evacuation for each populated city)
__global__ void evacuation_kernel(int *dist_flat, int *prev_flat, long long *city, long long *population, int num_populated_cities, long long *shelter_cities, long long *shelter_caps, int num_shelters, int max_dist_elderly, int num_cities, long long *path_size, long long *paths, long long *num_drops, long long *drops, int *roads)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_populated_cities)
        return;         // Return if thread index exceeds number of populated cities
    int origin = (int)city[tid];
    long long prime = population[2 * tid], elderly = population[2 * tid + 1];
    path_size[tid] = 0;
    num_drops[tid] = 0;
    int local_path[MAX_PATH];
    bool visited[100000] = {false};
    long long full_path[MAX_PATH];
    int full_len = 0;

    while (prime > 0 || elderly > 0)    //we will keep evacuating until all people are moved
    {
        // finding best shelter
        int best_s = findBestShelter(origin, visited, shelter_caps, dist_flat, num_shelters, num_cities, max_dist_elderly, elderly, prime);
        if (best_s < 0)
            break;
            
        visited[best_s] = true;
        int best_d = dist_flat[best_s * num_cities + origin];

        // constructing path from origin to shelter
        int plen = constructPath(origin, (int)shelter_cities[best_s], prev_flat, best_s, num_cities, local_path, MAX_PATH);
        
        // checking if a valid path is found
        if (local_path[plen-1] != (int)shelter_cities[best_s])
            continue;
            
        // merge paths
        full_len = mergePaths(local_path, plen, full_path, full_len);
        
        // Process of dropping people at shelter
        processDrop(tid, shelter_caps, best_s, best_d, max_dist_elderly, &elderly, &prime, num_drops, drops, shelter_cities[best_s], origin);
                  
        // Move origin to shelter for next iteration
        origin = (int)shelter_cities[best_s];
    }
    path_size[tid] = full_len;
    // writing the full path into global memory
    for (int i = 0; i < full_len; ++i)
        paths[tid * MAX_PATH + i] = full_path[i];
}

// holds all input parameters and host arrays
struct HostInput
{
    long long num_cities, num_roads, num_shelters, num_populated_cities;
    int max_distance_elderly;
    int *roads;
    long long *shelter_cities, *shelter_caps, *city, *population;
};

// holds all GPU buffers including flat dist/prev arrays
struct DeviceBuffers
{
    int *d_roads, *d_dist, *d_prev;
    long long *d_shelter_cities, *d_shelter_caps, *d_city, *d_population;
    long long *d_path_size, *d_paths_flat, *d_num_drops, *d_drops_flat;
};

// holds host-side result arrays
struct HostOutput
{
    long long *path_size;
    long long **paths;
    long long *num_drops;
    long long ***drops;
};

//read input file into HostInput
void readInput(const char *fname, HostInput &in)
{
    ifstream fin(fname);
    if (!fin)
    {
        cerr << "Cannot open " << fname << "\n";
        exit(1);
    }
    fin >> in.num_cities >> in.num_roads;
    in.roads = new int[in.num_roads * 4];
    for (int i = 0; i < in.num_roads; i++)
        fin >> in.roads[4 * i] >> in.roads[4 * i + 1] >> in.roads[4 * i + 2] >> in.roads[4 * i + 3];
    fin >> in.num_shelters;
    in.shelter_cities = new long long[in.num_shelters];
    in.shelter_caps = new long long[in.num_shelters];
    for (int i = 0; i < in.num_shelters; i++)
        fin >> in.shelter_cities[i] >> in.shelter_caps[i];
    fin >> in.num_populated_cities;
    in.city = new long long[in.num_populated_cities];
    in.population = new long long[in.num_populated_cities * 2];
    for (int i = 0; i < in.num_populated_cities; i++)
        fin >> in.city[i] >> in.population[2 * i] >> in.population[2 * i + 1];
    fin >> in.max_distance_elderly;
}

// allocate GPU buffers and copy host data to device
void setupDeviceMemory(const HostInput &in, DeviceBuffers &d)
{
    cudaMalloc(&d.d_roads, in.num_roads * 4 * sizeof(int));
    cudaMalloc(&d.d_shelter_cities, in.num_shelters * sizeof(long long));
    cudaMalloc(&d.d_shelter_caps, in.num_shelters * sizeof(long long));
    cudaMalloc(&d.d_city, in.num_populated_cities * sizeof(long long));
    cudaMalloc(&d.d_population, in.num_populated_cities * 2 * sizeof(long long));
    cudaMalloc(&d.d_dist, in.num_shelters * in.num_cities * sizeof(int));
    cudaMalloc(&d.d_prev, in.num_shelters * in.num_cities * sizeof(int));
    cudaMalloc(&d.d_path_size, in.num_populated_cities * sizeof(long long));
    cudaMalloc(&d.d_paths_flat, in.num_populated_cities * MAX_PATH * sizeof(long long));
    cudaMalloc(&d.d_num_drops, in.num_populated_cities * sizeof(long long));
    cudaMalloc(&d.d_drops_flat, in.num_populated_cities * MAX_DROPS * 3 * sizeof(long long));
    cudaMemcpy(d.d_roads, in.roads, in.num_roads * 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d.d_shelter_cities, in.shelter_cities, in.num_shelters * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d.d_shelter_caps, in.shelter_caps, in.num_shelters * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d.d_city, in.city, in.num_populated_cities * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d.d_population, in.population, in.num_populated_cities * 2 * sizeof(long long), cudaMemcpyHostToDevice);
}

//compute shortest paths per shelter via Bellman–Ford
void computeAllPairsDistances(const HostInput &in, DeviceBuffers &d)
{
    int bs = 256, g = (in.num_shelters + bs - 1) / bs;
    init_kernel<<<g, bs>>>(d.d_dist, d.d_prev, d.d_shelter_cities, in.num_shelters, in.num_cities);
    cudaDeviceSynchronize();
    int total = in.num_shelters * in.num_roads;
    g = (total + bs - 1) / bs;
    for (int i = 0; i < in.num_cities - 1; i++)
        relax_edges<<<g, bs>>>(d.d_dist, d.d_prev, d.d_roads, in.num_roads, in.num_cities, d.d_shelter_cities, in.num_shelters);
    cudaDeviceSynchronize();
}

//launch evacuation kernel for populated cities
void simulateEvacuation(const HostInput &in, DeviceBuffers &d)
{
    int bs = 1024, g = (in.num_populated_cities + bs - 1) / bs;
    evacuation_kernel<<<g, bs>>>(d.d_dist, d.d_prev, d.d_city, d.d_population, in.num_populated_cities, d.d_shelter_cities, d.d_shelter_caps, in.num_shelters, in.max_distance_elderly, in.num_cities, d.d_path_size, d.d_paths_flat, d.d_num_drops, d.d_drops_flat, d.d_roads);
    cudaDeviceSynchronize();
}

//copy results from device into HostOutput arrays
void copyResultsBack(const HostInput &in, DeviceBuffers &d, HostOutput &out)
{
    out.path_size = new long long[in.num_populated_cities];
    long long *pf = new long long[in.num_populated_cities * MAX_PATH];
    out.num_drops = new long long[in.num_populated_cities];
    long long *df = new long long[in.num_populated_cities * MAX_DROPS * 3];
    cudaMemcpy(out.path_size, d.d_path_size, in.num_populated_cities * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(pf, d.d_paths_flat, in.num_populated_cities * MAX_PATH * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(out.num_drops, d.d_num_drops, in.num_populated_cities * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(df, d.d_drops_flat, in.num_populated_cities * MAX_DROPS * 3 * sizeof(long long), cudaMemcpyDeviceToHost);
    out.paths = new long long *[in.num_populated_cities];
    out.drops = new long long **[in.num_populated_cities];
    for (int i = 0; i < in.num_populated_cities; i++)
    {
        out.paths[i] = &pf[i * MAX_PATH];
        out.drops[i] = new long long *[out.num_drops[i]];
        for (int j = 0; j < out.num_drops[i]; j++)
            out.drops[i][j] = &df[i * MAX_DROPS * 3 + j * 3];
    }
}

// freeing all host and device memory
void cleanup(const HostInput &in, HostOutput &out, DeviceBuffers &d)
{
    delete[] in.roads;
    delete[] in.shelter_cities;
    delete[] in.shelter_caps;
    delete[] in.city;
    delete[] in.population;
    delete[] out.path_size;
    for (int i = 0; i < in.num_populated_cities; i++)
        delete[] out.drops[i];
    delete[] out.paths;
    delete[] out.drops;
    cudaFree(d.d_roads);
    cudaFree(d.d_shelter_cities);
    cudaFree(d.d_shelter_caps);
    cudaFree(d.d_city);
    cudaFree(d.d_population);
    cudaFree(d.d_dist);
    cudaFree(d.d_prev);
    cudaFree(d.d_path_size);
    cudaFree(d.d_paths_flat);
    cudaFree(d.d_num_drops);
    cudaFree(d.d_drops_flat);
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <in> <out>\n";
        return 1;
    }
    HostInput in{};
    DeviceBuffers d{};
    HostOutput out{};
    readInput(argv[1], in);
    auto t0 = chrono::high_resolution_clock::now();
    setupDeviceMemory(in, d);
    computeAllPairsDistances(in, d);
    simulateEvacuation(in, d);
    cudaDeviceSynchronize();
    auto t1 = chrono::high_resolution_clock::now();
    cout << chrono::duration<double>(t1 - t0).count() << " s\n";
    copyResultsBack(in, d, out);
    
    ofstream outfile(argv[2]); // Read input file from command-line argument
    if (!outfile) {
        cerr << "Error: Cannot open file " << argv[2] << "\n";
        return 1;
    }

    for(long long i = 0; i < in.num_populated_cities; i++){
        long long currentPathSize = out.path_size[i];
        for(long long j = 0; j < currentPathSize; j++){
            outfile << out.paths[i][j] << " ";
        }
        outfile << "\n";
    }

    for(long long i = 0; i < in.num_populated_cities; i++){
        long long currentDropSize = out.num_drops[i];
        for(long long j = 0; j < currentDropSize; j++){
            for(int k = 0; k < 3; k++){
                outfile << out.drops[i][j][k] << " ";
            }
        }
        outfile << "\n";
    }
    
    cleanup(in, out, d);
    return 0;
}
