
# include<stdlib.h>
#include <iostream>
#include <vector>
# include<ctime>
# include<random> 
#include<string>
#include <fstream>
#include <sstream>

struct StockData{
    std::string Datetime;
    std::string stockId;
    int volume;
    double  price;
};

StockData parse_stock_data(const std::string& line) {

    StockData data;
    std::istringstream iss(line);
    std::string token;

    std::getline(iss, token, ',');
    data.stockId = token;

    std::getline(iss, token, ',');
    data.Datetime = token;

    std::getline(iss, token, ',');
    data.volume = std::stoi(token);

    std::getline(iss, token, ',');
    data.price = std::stod(token);
    return data;
}
int main()
{
    std::ifstream file("/home/wxcwxc/ctensorRt/C++Project/Cquant/stock.csv");
    if (!file.is_open()){
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }

    std::vector<StockData> stock_dataList;
    std::string line;
    std::string header;
    std::getline(file, header);

    while( std::getline(file,line)){
        StockData data = parse_stock_data(line);
        stock_dataList.push_back(data);
    }

    double totalPrice= 0; 
    int  count = 0; 

    for (auto & data :stock_dataList){
        
        totalPrice +=data.price;
        count ++;

        std::cout <<"StockID : "<<data.stockId <<",Time :"<<data.Datetime << ",Volume :" <<
        data.volume <<", Price :"<<data.price <<std::endl; 
    }
    std::cout <<"avgPrice :" << totalPrice / count <<std::endl;
    file.close();
    return 0;
}