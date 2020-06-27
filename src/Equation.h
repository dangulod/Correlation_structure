#ifndef EQUATION_H__
#define EQUATION_H__

class CorMatrix;

#include <armadillo>
#include <vector>
#include <string>
#include "FactorCorrelation.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

namespace pt = boost::property_tree;

using std::string;
using std::vector;

struct Weight
{
    string name;
    double weight;
    bool optim;
};

class Equation : public vector<Weight>
{
private:
    string name;

public:
    Equation() = delete;
    Equation(string name, vector<Weight> sensib);
    Equation(const Equation & value) = default;
    Equation(Equation && value) = default;
    Equation operator=(Equation value);
	~Equation() = default;

    pt::ptree to_ptree();
    static Equation from_ptree(pt::ptree & value);

    double get_weight(string factor);
    double get_weight(size_t n);
    string get_name();

    void set_weights(size_t pos, double value);
    void set_weights(arma::vec value);

    double R2(CorMatrix &cor);
};



#endif // !EQUATION_H__
