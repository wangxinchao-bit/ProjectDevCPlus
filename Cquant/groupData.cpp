#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <thread>
#include <filesystem>
#include <queue>

// 互斥量用于保护共享资源
std::mutex mtx;

// 读取文件内容并根据股票代码进行分组
void read_and_group_stock_data(const std::string& filename, std::unordered_map<std::string, std::string>& stock_data) {
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        // 逐行读取文件内容
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string stock_id;
            std::getline(iss, stock_id, ','); // 获取股票代码
            // 临界区，需要保护共享资源
            std::lock_guard<std::mutex> lock(mtx);
            // 将股票数据按照股票代码进行分组
            stock_data[stock_id] += line + "\n";
        }
        file.close();
    } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
}

// 写入股票数据到单独的文件中
void write_stock_data_to_file(const std::string& stock_id, const std::string& data) {
    std::ofstream file("stock_files/" + stock_id + ".csv");
    if (file.is_open()) {
        file << data;
        file.close();
    } else {
        std::cerr << "Failed to create file: " << stock_id + ".csv" << std::endl;
    }
}

int main() {
    std::unordered_map<std::string, std::string> stock_data;

    // 读取包含所有股票数据的文件并根据股票代码进行分组
    read_and_group_stock_data("/home/wxcwxc/ctensorRt/C++Project/Cquant/data.csv", stock_data);

    // 创建一个文件夹来存储单只股票的文件
    std::filesystem::create_directory("stock_files");

    std::queue<std::pair<std::string, std::string>> work_queue;
    std::mutex queue_mtx;

    // 将分组后的股票数据放入工作队列
    for (const auto& pair : stock_data) {
        work_queue.push(pair);
    }

    // 最多只启动5个线程来处理工作队列中的任务
    std::vector<std::thread> threads;
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&work_queue, &queue_mtx]() {
            while (true) {
                std::pair<std::string, std::string> work_item;
                {
                    // 从工作队列中获取任务
                    std::lock_guard<std::mutex> lock(queue_mtx);
                    if (work_queue.empty()) {
                        break;
                    }
                    work_item = work_queue.front();
                    work_queue.pop();
                }
                // 处理任务
                write_stock_data_to_file(work_item.first, work_item.second);
            }
        });
    }

    // 等待所有线程结束
    for (auto& thread : threads) {
        thread.join();
    }

    return 0;
}
