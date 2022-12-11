#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <functional>

bool g_bRun = true;

// This is the function that will be executed by each thread in the thread pool.
// It will continuously run and check if there are any jobs in the queue. If there
// are, it will take a job from the queue and execute it.
void thread_function(std::mutex& queue_mutex, std::queue<std::function<void()>>& jobs)
{
    while (g_bRun && !jobs.empty())
    {
        // Lock the queue mutex so we can access the queue safely.
        std::lock_guard<std::mutex> lock(queue_mutex);

        // Check if there are any jobs in the queue.
        if (jobs.empty())
        {
            // If there are no jobs, we can just continue to the next iteration
            // of the loop and check again.
            continue;
        }

        // If there are jobs in the queue, we take the first one and execute it.
        auto job = jobs.front();
        jobs.pop();
        job();
    }
}

int calculate_prime_sum(int start, int end)
{
    int sum = 0;

    // Loop over the range of numbers and check if each number is prime.
    for (int i = start; i <= end; ++i)
    {
        bool is_prime = true;

        // Check if the number is prime by trying to divide it by all numbers
        // from 2 to the square root of the number.
        for (int j = 2; j * j <= i; ++j)
        {
            if (i % j == 0)
            {
                // If the number is divisible by any of these numbers, it is not prime.
                is_prime = false;
                break;
            }
        }

        // If the number is prime, add it to the sum.
        if (is_prime)
        {
            sum += i;
        }
    }

    return sum;
}

int main()
{
    // We create a mutex to synchronize access to the queue of jobs.
    std::mutex queue_mutex;

    // This is the queue of jobs. Each job is a function that can be executed
    // by a thread in the thread pool.
    std::queue<std::function<void()>> jobs;

    // This is the vector of threads in the thread pool.
    std::vector<std::thread> threads;

    // Here, we create four threads in the thread pool.
     const size_t numThreads = std::thread::hardware_concurrency();
    threads.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i)
    {
        threads.emplace_back(thread_function, std::ref(queue_mutex), std::ref(jobs));
    }

    // Now we can add some jobs to the queue. In this example, we just add a few
    // simple jobs that print a message to the console.
    for (int i = 0; i < 5000; ++i)
    {
        // We use a lambda function to create a job that prints a message to the console.
        auto job = [i]()
        {
            std::cout << "Hello from thread " << std::this_thread::get_id() << " i: " << i << std::endl;
			double result = std::sqrt(i*i*i)*std::sqrt(calculate_prime_sum(1, 1000000 * i));
        };

        // We add the job to the queue.
        std::lock_guard<std::mutex> lock(queue_mutex);
        jobs.push(job);
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
	g_bRun = false;
    // Wait for all threads to finish before exiting the program.
    for (auto& thread : threads)
    {
        thread.join();
    }

	std::cout << "finished!" << std::endl;
    getchar();
    return 0;
}
