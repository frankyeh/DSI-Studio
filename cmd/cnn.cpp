#include <QApplication>
#include <QFileInfo>
#include "program_option.hpp"
#include "tipl/tipl.hpp"
#include "gzip_interface.hpp"

int cnn(void)
{
    std::string train_file_name = po.get("train");
    tipl::ml::network_data<float,unsigned char> nn_data,nn_test;
    if(!nn_data.load_from_file<gz_istream>(train_file_name.c_str()))
    {
        std::cout << "Cannot load training data at " << train_file_name << std::endl;
        return 0;
    }
    if(po.has("test"))
    {
        std::string test_file_name = po.get("test");
        if(!nn_test.load_from_file<gz_istream>(test_file_name.c_str()))
        {
            std::cout << "Cannot load testing data at " << test_file_name << std::endl;
            return 0;
        }
    }
    if(po.get("test_from_train",0.1f) != 0.0f)
    {
        std::cout << "Extract test dataset from training dataset" << std::endl;
        nn_data.sample_test_from(nn_test,po.get("test_from_train",0.1f));
    }

    std::string network = po.get("network");
    tipl::ml::network nn;
    if(network.find(".txt") != std::string::npos)
    {
        std::ifstream in(network.c_str());
        in >> network;
    }

    if(!(nn << network))
    {
        std::cout << "Invalid network: " << nn.error_msg << std::endl;
        return 0;
    }

    nn.learning_rate = po.get("learning_rate",0.001f);
    nn.w_decay_rate = po.get("w_decay_rate",0.0001f);
    nn.b_decay_rate = po.get("b_decay_rate",0.2f);
    nn.momentum = po.get("momentum",0.9f);
    nn.batch_size = po.get("batch_size",64);
    nn.epoch = po.get("epoch",20);
    nn.repeat = po.get("repeat",10);

    std::cout << "learning rate=" << nn.learning_rate << std::endl;
    std::cout << "weight decay=" << nn.w_decay_rate << std::endl;
    std::cout << "bias decay=" << nn.b_decay_rate << std::endl;
    std::cout << "momentum=" << nn.momentum << std::endl;
    std::cout << "batch size=" << nn.batch_size << std::endl;
    std::cout << "epoch=" << nn.epoch << std::endl;

    auto on_enumerate_epoch = [&](){
        if(!nn_test.empty())
            std::cout << "testing error:" << nn.test_error(nn_test.data,nn_test.data_label) << "%" << std::endl;
        std::cout << "training error:" << nn.get_training_error() << "%" << std::endl;
        };
    nn.initialize_training();
    bool terminated = false;
    nn.train(nn_data,terminated, on_enumerate_epoch);

    if(po.has("output_nn"))
    {
        std::string output = po.get("output");
        if(!output.empty())
            nn.save_to_file<gz_ostream>(output.c_str());
    }
    if(po.has("output_error"))
    {
        std::string prefix = po.get("output_error");
        std::ostringstream out;
        out << prefix << std::setfill('0') << std::setw(4) <<
               int(nn.test_error(nn_test.data,nn_test.data_label)*1000.0) << ".txt";
        std::ofstream out2(out.str().c_str());
        out2 << network;
    }

    return 0;
}
