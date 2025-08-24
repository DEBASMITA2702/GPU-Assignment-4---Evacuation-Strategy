#include <iostream>
#include <fstream>
#include <cuda.h>

// these connstants are used to define the size of various arrays in the problem
#define MAX_DROPS 1000
#define MAX_PATH 1000
#define MAX_ATTEMPTS 10000
#define LOCK_FREE -1

using namespace std;

// arrays to be used for solving the road contention
__device__ double road_availability_time[100000];
__device__ long long road_last_user[100000];

// device function to perform atomicSub for long long datatype
__device__ long long atomicSubLL(long long* address, long long val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed - val);
    } while (assumed != old);
    return (long long)old;
}

// device function to perform atomicAdd for long long datatype
__device__ long long atomicAddLL(long long* address, long long val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed + val);
    } while (assumed != old);
    return (long long)old;
}

// kernel to initialize the distance and the parent arrays for the bellman ford execution
__global__ void init_kernel(int **dist, int **prev, long long *shelters,int num_shelters, int num_cities) {
    int shelter_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (shelter_idx >= num_shelters) return;

    int shelter = shelters[shelter_idx];
    int *dist_shelter = dist[shelter_idx];
    int *prev_shelter = prev[shelter_idx];
    
    for (int i = 0; i < num_cities; i++) {
        if (i == shelter)
            dist_shelter[i] = 0;
        else
            dist_shelter[i] = INT_MAX;
        
        prev_shelter[i] = -1;
    }
}


// kernel to run the edge relaxation of all the edges during one iteration of the Bellman ford
__global__ void relax_edges(int **dist, int **prev, int *roads,int num_roads, int num_cities,long long *shelters, int num_shelters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tasks = num_shelters * num_roads;

    if (tid >= total_tasks) return;

    int shelter_idx = tid / num_roads;
    int edge_idx = tid % num_roads;

    int *dist_shelter = dist[shelter_idx];
    int *prev_shelter = prev[shelter_idx];

    int u = roads[4 * edge_idx];
    int v = roads[4 * edge_idx + 1];
    int length = roads[4 * edge_idx + 2];

    // run the edge relaxation in both directions (u to v and v to u) since the graph is unidrected
    // no need of any synchronization as each shelter has its own set of threads to execute this kernel
    // also the dist and prev arrays have different rows for each of the shelter
    if (dist_shelter[u] != INT_MAX && dist_shelter[u] + length < dist_shelter[v]) {
        dist_shelter[v] = dist_shelter[u] + length;
        prev_shelter[v] = u;
    }

    if (dist_shelter[v] != INT_MAX && dist_shelter[v] + length < dist_shelter[u]) {
        dist_shelter[u] = dist_shelter[v] + length;
        prev_shelter[u] = v;
    }
}

// kernel to run the evacutation process (here each thread takes care of the evacuation for each populated city)
__global__ void evacuation_kernel(int **dist, int **prev, long long *city, long long *population, int num_populated_cities, long long *shelter_cities, long long *shelter_caps, int num_shelters, int max_dist_elderly, int num_cities, long long *path_size, long long *paths, long long *num_drops, long long *drops, int *roads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_populated_cities) return;

    int origin_city = city[tid];
    long long prime_left = population[2 * tid];
    long long elderly_left = population[2 * tid + 1];

    path_size[tid] = 0;
    num_drops[tid] = 0;

    int path[MAX_PATH];
    bool visited_shelters[100000] = {false};

    while (prime_left > 0 || elderly_left > 0) {
        int best_shelter = -1;
        int best_dist = INT_MAX;

        for (int s = 0; s < num_shelters; s++) {
            if (visited_shelters[s]) continue;
            int d = dist[s][origin_city];
            if (d == INT_MAX || shelter_caps[s] <= 0) continue;

            bool elderly_can_reach = (elderly_left == 0 || d <= max_dist_elderly);
            if (!elderly_can_reach && prime_left == 0) continue;

            if (d < best_dist) {
                best_dist = d;
                best_shelter = s;
            }
        }

        if (best_shelter == -1) break;
        visited_shelters[best_shelter] = true;

        // path construction from prev array
        int plen = 0, curr = origin_city;
        while (curr != -1 && plen < MAX_PATH) {
            path[plen++] = curr;
            if (curr == shelter_cities[best_shelter]) break;
            curr = prev[best_shelter][curr];
        }
        if (curr != shelter_cities[best_shelter]) continue;

        for (int i = 0; i < plen; i++)
            paths[tid * MAX_PATH + i] = path[i];
        path_size[tid] = plen;

        
        long long dropped_elderly = 0;
        if (best_dist <= max_dist_elderly) {
            dropped_elderly = elderly_left;
        }

        long long available_cap = atomicAddLL(&shelter_caps[best_shelter], 0);
        long long total_to_drop = prime_left + dropped_elderly;
        long long can_drop = min(total_to_drop, available_cap);

        if (can_drop <= 0) continue;

        atomicSubLL(&shelter_caps[best_shelter], can_drop);

        dropped_elderly = min(can_drop, dropped_elderly);
        elderly_left -= dropped_elderly;

        long long dropped_prime = can_drop - dropped_elderly;
        prime_left -= dropped_prime;

        // add the drop to the result
        int drop_idx = atomicAddLL(&num_drops[tid], 1);
        drops[tid * MAX_DROPS * 3 + drop_idx * 3 + 0] = shelter_cities[best_shelter];
        drops[tid * MAX_DROPS * 3 + drop_idx * 3 + 1] = dropped_elderly;
        drops[tid * MAX_DROPS * 3 + drop_idx * 3 + 2] = dropped_prime;

        if (elderly_left > 0 && best_dist > max_dist_elderly) {
            int extra_drop_idx = atomicAddLL(&num_drops[tid], 1);
            drops[tid * MAX_DROPS * 3 + extra_drop_idx * 3 + 0] = origin_city;
            drops[tid * MAX_DROPS * 3 + extra_drop_idx * 3 + 1] = elderly_left;
            drops[tid * MAX_DROPS * 3 + extra_drop_idx * 3 + 2] = 0;
            elderly_left = 0;
        }

        origin_city = shelter_cities[best_shelter];
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    ifstream infile(argv[1]); // Read input file from command-line argument
    if (!infile) {
        cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }
    
    long long num_cities;
    infile >> num_cities;

    long long num_roads;
    infile >> num_roads;

    // Store roads as a flat array: [u1, v1, length1, capacity1, u2, v2, length2, capacity2, ...]
    int *roads = new int[num_roads * 4]; 

    for (int i = 0; i < num_roads; i++) {
        infile >> roads[4 * i] >> roads[4 * i + 1] >> roads[4 * i + 2] >> roads[4 * i + 3];
    }

    int num_shelters;
    infile >> num_shelters;

    // Store shelters separately
    long long *shelter_city = new long long[num_shelters];
    long long *shelter_capacity = new long long[num_shelters];

    for (int i = 0; i < num_shelters; i++) {
        infile >> shelter_city[i] >> shelter_capacity[i];
    }

    int num_populated_cities;
    infile >> num_populated_cities;

    // Store populated cities separately
    long long *city = new long long[num_populated_cities];
    long long *pop = new long long[num_populated_cities * 2]; // Flattened [prime-age, elderly] pairs

    for (long long i = 0; i < num_populated_cities; i++) {
        infile >> city[i] >> pop[2 * i] >> pop[2 * i + 1];
    }

    int max_distance_elderly;
    infile >> max_distance_elderly;

    infile.close();

    
    // set your answer to these variables
    long long *path_size;
    long long **paths;
    long long *num_drops;
    long long ***drops;
	

	// my code starts here

    // declaration of device arrays and allocating memory
	long long *d_path_size, *d_paths_flat;
	long long *d_num_drops, *d_drops_flat;

	cudaMalloc(&d_path_size, num_populated_cities * sizeof(long long));
	cudaMalloc(&d_paths_flat, num_populated_cities * MAX_PATH * sizeof(long long));
	cudaMalloc(&d_num_drops, num_populated_cities * sizeof(long long));
	cudaMalloc(&d_drops_flat, num_populated_cities * MAX_DROPS * 3 * sizeof(long long));
	
	int *d_roads;
	long long *d_shelters, *d_shelter_capacity;
	long long *d_city, *d_pop;
    int **d_dist, **d_prev;

    cudaMalloc(&d_roads, num_roads * 4 * sizeof(int));
    cudaMalloc(&d_shelters, num_shelters * sizeof(long long));
	cudaMalloc(&d_shelter_capacity, num_shelters * sizeof(long long));
	cudaMalloc(&d_city, num_populated_cities * sizeof(long long));
	cudaMalloc(&d_pop, num_populated_cities * 2 * sizeof(long long));
    cudaMemcpy(d_roads, roads, num_roads * 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shelters, shelter_city, num_shelters * sizeof(long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d_shelter_capacity, shelter_capacity, num_shelters * sizeof(long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d_city, city, num_populated_cities * sizeof(long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pop, pop, num_populated_cities * 2 * sizeof(long long), cudaMemcpyHostToDevice);
	
	int **h_dist = (int **)malloc(num_shelters * sizeof(int *));
	for (int i = 0; i < num_shelters; i++) {
		cudaMalloc(&h_dist[i], num_cities * sizeof(int));
	}
    cudaMalloc(&d_dist, num_shelters * sizeof(int *));
	cudaMemcpy(d_dist, h_dist, num_shelters * sizeof(int *), cudaMemcpyHostToDevice);
	int **h_prev = (int **)malloc(num_shelters * sizeof(int *));
	for (int i = 0; i < num_shelters; i++) {
		cudaMalloc(&h_prev[i], num_cities * sizeof(int));
	}
	cudaMalloc(&d_prev, num_shelters * sizeof(int *));
	cudaMemcpy(d_prev, h_prev, num_shelters * sizeof(int *), cudaMemcpyHostToDevice);
	
	free(h_dist);
	free(h_prev);


    // kernel calls and actual execution
    int block_size = 1024;
	
    int blocks = ceil((float)num_shelters / block_size);
	init_kernel<<<blocks, block_size>>>(d_dist, d_prev, d_shelters, num_shelters, num_cities);

	int total_tasks = num_shelters * num_roads;
	blocks = ceil((float)total_tasks / block_size);
	for (int iter = 0; iter < num_cities - 1; iter++) {
		relax_edges<<<blocks, block_size>>>(d_dist, d_prev, d_roads, num_roads,num_cities, d_shelters, num_shelters);
	}
	
	blocks = ceil((float)num_populated_cities / block_size);
	evacuation_kernel<<<blocks, block_size>>>(d_dist, d_prev, d_city, d_pop, num_populated_cities, d_shelters, d_shelter_capacity, num_shelters, max_distance_elderly, num_cities, d_path_size, d_paths_flat, d_num_drops, d_drops_flat, d_roads);
	
	cudaDeviceSynchronize();
	
	
	// copy back the resutls;	
	path_size = new long long[num_populated_cities];
	long long *paths_flat = new long long[num_populated_cities * MAX_PATH];
	num_drops = new long long[num_populated_cities];
	long long *drops_flat = new long long[num_populated_cities * MAX_DROPS * 3];

	cudaMemcpy(path_size, d_path_size, num_populated_cities * sizeof(long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(paths_flat, d_paths_flat, num_populated_cities * MAX_PATH * sizeof(long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(num_drops, d_num_drops, num_populated_cities * sizeof(long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(drops_flat, d_drops_flat, num_populated_cities * MAX_DROPS * 3 * sizeof(long long), cudaMemcpyDeviceToHost);

	// store the results back to the 2d and 3d arrays
	paths = new long long*[num_populated_cities];
    drops = new long long**[num_populated_cities];
	for (int i = 0; i < num_populated_cities; ++i) {
		paths[i] = &paths_flat[i * MAX_PATH];
        drops[i] = new long long*[num_drops[i]];
		for (int j = 0; j < num_drops[i]; ++j)
			drops[i][j] = &drops_flat[i * MAX_DROPS * 3 + j * 3];
    }
	
	cudaFree(d_roads);
	cudaFree(d_shelters);
	cudaFree(d_shelter_capacity);
	cudaFree(d_city);
	cudaFree(d_pop);
	cudaFree(d_dist);
	cudaFree(d_prev);
	cudaFree(d_path_size);
	cudaFree(d_paths_flat);
	cudaFree(d_num_drops);
	cudaFree(d_drops_flat);
	// my code ends
	
	
    ofstream outfile(argv[2]); // Read input file from command-line argument
    if (!outfile) {
        cerr << "Error: Cannot open file " << argv[2] << "\n";
        return 1;
    }

    for(long long i = 0; i < num_populated_cities; i++){
        long long currentPathSize = path_size[i];
        for(long long j = 0; j < currentPathSize; j++){
            outfile << paths[i][j] << " ";
        }
        outfile << "\n";
    }

    for(long long i = 0; i < num_populated_cities; i++){
        long long currentDropSize = num_drops[i];
        for(long long j = 0; j < currentDropSize; j++){
            for(int k = 0; k < 3; k++){
                outfile << drops[i][j][k] << " ";
            }
        }
        outfile << "\n";
    }

    return 0;
}

