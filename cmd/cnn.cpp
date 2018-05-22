#include <QApplication>
#include <QFileInfo>
#include "program_option.hpp"
#include "tipl/tipl.hpp"
#include "gzip_interface.hpp"
bool train_cnn(std::string network,
               tipl::ml::network& nn,
               tipl::ml::network_data<float,unsigned char>& nn_data,
               tipl::ml::network_data<float,unsigned char>& nn_test,
               float& test_error,
               float& train_error)
{

    if(!(nn << network))
    {
        std::cout << "Invalid network: " << nn.error_msg << std::endl;
        return false;
    }
    std::cout << "training network=" << network << std::endl;
    nn.learning_rate = po.get("learning_rate",0.01f);
    nn.w_decay_rate = po.get("w_decay_rate",0.0001f);
    nn.b_decay_rate = po.get("b_decay_rate",0.2f);
    nn.momentum = po.get("momentum",0.05f);
    nn.batch_size = po.get("batch_size",64);
    nn.epoch = po.get("epoch",2000);

    std::cout << "learning rate=" << nn.learning_rate << std::endl;
    std::cout << "weight decay=" << nn.w_decay_rate << std::endl;
    std::cout << "bias decay=" << nn.b_decay_rate << std::endl;
    std::cout << "momentum=" << nn.momentum << std::endl;
    std::cout << "batch size=" << nn.batch_size << std::endl;
    std::cout << "epoch=" << nn.epoch << std::endl;

    auto on_enumerate_epoch = [&](){
        if(po.has("rotation"))
            nn_data.rotate_permute();
        if(!nn_test.empty())
            std::cout << "testing error:" << (test_error = nn.test_error(nn_test.data,nn_test.data_label)) << "%" << std::endl;
        std::cout << "training error:" << (train_error = nn.get_training_error()) << "%" << std::endl;
        };
    nn.init_weights();
    bool terminated = false;
    nn.train(nn_data,terminated, on_enumerate_epoch);
    return true;
}

int cnn(void)
{
    std::string train_file_name = po.get("train");
    tipl::ml::network_data<float,unsigned char> nn_data,nn_test;
    if(!nn_data.load_from_file<gz_istream>(train_file_name.c_str()))
    {
        std::cout << "Cannot load training data at " << train_file_name << std::endl;
        return 0;
    }
    std::cout << "A total of "<< nn_data.size() << " training data are loaded." << std::endl;
    if(po.has("test"))
    {
        std::string test_file_name = po.get("test");
        if(!nn_test.load_from_file<gz_istream>(test_file_name.c_str()))
        {
            std::cout << "Cannot load testing data at " << test_file_name << std::endl;
            return 0;
        }
    }
    std::cout << "A total of "<< nn_test.size() << " testing data are loaded." << std::endl;

    std::vector<std::string> network_list;
    {
        std::string network = po.get("network");
        std::ifstream in(network.c_str());
        if(!in)
        {
            std::cout << "Cannot open " << network << std::endl;
            return 0;
        }
        std::string line;
        while(std::getline(in,line))
        if(line.size() > 5)
            network_list.push_back(line);
    }
    std::cout << "A network list is loaded with " << network_list.size() << " networks." << std::endl;

    // run network list
    if(network_list.size() > 1)
    {       
        int thread_id = po.get("thread_id",0);
        int thread_count = po.get("thread_count",1);
        std::string output_name = po.get("network")+"."+std::to_string(thread_id)+".txt";
        int start_count = 0;
        {
            std::ifstream in2(output_name.c_str());
            std::string line;
            while(std::getline(in2,line))
                ++start_count;
        }
        std::ofstream out(output_name.c_str(),std::ios::app);
        for(int i = thread_id;i < network_list.size();i += thread_count)
            {
                if(start_count)
                {
                    std::cout << "Skipping network:" << network_list[i] << std::endl;
                    --start_count;
                    continue;
                }
                float test_error = 0.0,train_error = 0.0;
                tipl::ml::network nn;
                if(!train_cnn(network_list[i],nn,nn_data,nn_test,test_error,train_error))
                    continue;
                std::cout << "Training finished" << std::endl;
                std::cout << test_error << "\t" << train_error << "\t" << network_list[i] << std::endl;
                out << test_error << "\t" << train_error << "\t" << network_list[i] << std::endl;
            }
        return 0;
    }

    float test_error = 0.0,train_error = 0.0;
    tipl::ml::network nn;
    if(!train_cnn(network_list[0],nn,nn_data,nn_test,test_error,train_error))
        return 0;
    std::cout << "Training finished" << std::endl;
    std::cout << test_error << "," << train_error << "," << network_list[0] << std::endl;

    if(po.has("output_nn"))
        nn.save_to_file<gz_ostream>(po.get("output_nn").c_str());

    return 0;
}
