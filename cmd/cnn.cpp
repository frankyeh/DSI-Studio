#include <QApplication>
#include <QFileInfo>
#include "connectometry/group_connectometry_analysis.h"
bool train_cnn(tipl::ml::trainer& t,
               std::string network,
               tipl::ml::network& nn,
               tipl::ml::network_data<unsigned char>& nn_data_,
               tipl::ml::network_data<unsigned char>& nn_test_,
               float&,
               float& train_error)
{
    tipl::ml::network_data_proxy<unsigned char> nn_data = nn_data_;
    tipl::ml::network_data_proxy<unsigned char> nn_test = nn_test_;

    if(!(nn << network))
    {
        tipl::out() << "invalid network: " << nn.error_msg << std::endl;
        return false;
    }
    tipl::out() << "training network: " << network << std::endl;
    tipl::out() << "learning rate=" << t.learning_rate << std::endl;
    //tipl::out() << "weight decay=" << t.w_decay_rate << std::endl;
    tipl::out() << "momentum=" << t.momentum << std::endl;
    tipl::out() << "batch size=" << t.batch_size << std::endl;
    tipl::out() << "epoch=" << t.epoch << std::endl;

    auto on_enumerate_epoch = [&](){
        tipl::out() << "training error: " << (train_error = t.get_training_error()) << "%" << std::endl;
        };
    nn.init_weights();
    bool terminated = false;

    t.train(nn,nn_data,terminated, on_enumerate_epoch);
    return true;
}


int cnn(tipl::program_option<tipl::out>& po)
{
    std::shared_ptr<group_connectometry_analysis> gca(new group_connectometry_analysis);
    if(!gca->load_database(po.get("source").c_str()))
    {
        tipl::out() << "invalid database format" << std::endl;
        return 1;
    }


    if(!gca->handle->db.parse_demo(po.get("demo").c_str()))
    {
        tipl::out() << gca->handle->db.error_msg << std::endl;
        return 1;
    }




    std::string train_file_name = po.get("train");
    tipl::ml::network_data<unsigned char> nn_data,nn_test;
    if(!nn_data.load_from_file<tipl::io::gz_istream>(train_file_name.c_str()))
    {
        tipl::out() << "cannot load training data at " << train_file_name << std::endl;
        return 1;
    }
    tipl::out() << "a total of "<< nn_data.size() << " training data are loaded." << std::endl;
    if(po.has("test"))
    {
        std::string test_file_name = po.get("test");
        if(!nn_test.load_from_file<tipl::io::gz_istream>(test_file_name.c_str()))
        {
            tipl::out() << "cannot load testing data at " << test_file_name << std::endl;
            return 1;
        }
    }
    tipl::out() << "a total of "<< nn_test.size() << " testing data are loaded." << std::endl;

    std::vector<std::string> network_list;
    {
        std::string network = po.get("network");
        std::ifstream in(network.c_str());
        if(!in)
        {
            tipl::out() << "cannot open " << network << std::endl;
            return 1;
        }
        std::string line;
        while(std::getline(in,line))
        if(line.size() > 5)
            network_list.push_back(line);
    }
    tipl::out() << "a network list is loaded with " << network_list.size() << " networks." << std::endl;

    tipl::ml::trainer t;
    t.learning_rate = po.get("learning_rate",0.01f);
    //t.w_decay_rate = po.get("w_decay_rate",0.0f);
    t.momentum = po.get("momentum",0.5f);
    t.batch_size = po.get("batch_size",64);
    t.epoch = po.get("epoch",2000);

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
                    tipl::out() << "skipping network: " << network_list[i] << std::endl;
                    --start_count;
                    continue;
                }
                float test_error = 0.0,train_error = 0.0;
                tipl::ml::network nn;
                if(!train_cnn(t,network_list[i],nn,nn_data,nn_test,test_error,train_error))
                    continue;
                tipl::out() << "training finished" << std::endl;
                tipl::out() << test_error << "\t" << train_error << "\t" << network_list[i] << std::endl;
                out << test_error << "\t" << train_error << "\t" << network_list[i] << std::endl;
            }
        return 1;
    }

    float test_error = 0.0,train_error = 0.0;
    tipl::ml::network nn;
    if(!train_cnn(t,network_list[0],nn,nn_data,nn_test,test_error,train_error))
        return 1;
    tipl::out() << "training finished" << std::endl;
    tipl::out() << test_error << "," << train_error << "," << network_list[0] << std::endl;

    if(po.has("output_nn"))
        nn.save_to_file<tipl::io::gz_ostream>(po.get("output_nn").c_str());

    return 0;
}
