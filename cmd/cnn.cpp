#include <QApplication>
#include <QFileInfo>
#include "program_option.hpp"
#include "image/image.hpp"

int cnn(void)
{
    std::string train_file_name = po.get("train");
    image::ml::network_data<float,unsigned char> nn_data,nn_test;
    if(!nn_data.load_from_file(train_file_name.c_str()))
    {
        std::cout << "Cannot load training data at " << train_file_name << std::endl;
        return 0;
    }
    if(po.has("test"))
    {
        std::string test_file_name = po.get("test");
        if(!nn_test.load_from_file(test_file_name.c_str()))
        {
            std::cout << "Cannot load testing data at " << test_file_name << std::endl;
            return 0;
        }
    }
    std::string network = po.get("network");
    image::ml::network nn;

    if(!(nn << network))
    {
        std::cout << "Invalid network: " << nn.error_msg << std::endl;
        return 0;
    }
    nn.learning_rate = po.get("learning_rate",0.05f);
    nn.w_decay_rate = po.get("w_decay_rate",0.0001f);
    nn.b_decay_rate = po.get("b_decay_rate",0.2f);
    nn.momentum = po.get("momentum",0.9f);
    nn.batch_size = po.get("batch_size",64);
    nn.epoch = po.get("epoch",20);

    std::cout << "learning rate=" << nn.learning_rate << std::endl;
    std::cout << "weight decay=" << nn.w_decay_rate << std::endl;
    std::cout << "bias decay=" << nn.b_decay_rate << std::endl;
    std::cout << "momentum=" << nn.momentum << std::endl;
    std::cout << "batch size=" << nn.batch_size << std::endl;
    std::cout << "epoch=" << nn.epoch << std::endl;

    auto on_enumerate_epoch = [&](){
        if(!nn_test.is_empty())
            std::cout << "testing error:" << nn.test_error(nn_test.data,nn_test.data_label) << "%" << std::endl;
        std::cout << "training error:" << nn.get_training_error() << "%" << std::endl;
        };
    nn.initialize_training();
    bool terminated = false;
    nn.train(nn_data,terminated, on_enumerate_epoch);

    std::string output = po.get("output");
    if(!output.empty())
        nn.save_to_file(output.c_str());
    return 0;
}
