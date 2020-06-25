#ifndef FACTORCORRELATION_H__
#define FACTORCORRELATION_H__

class Equation;

#include <armadillo>
#include <vector>
#include <string>
#include <algorithm>
#include "Equation.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

namespace pt = boost::property_tree;

const static double EPSILON = 1e-9;

using std::string;
using std::vector;

/*
 * Crear una clase matriz con nombres
 * Cambiar Factor Correlation
 *
 */


class CorMatrix
{
    vector<string> factors;
    arma::mat cor;

public:
    CorMatrix() = delete;
    CorMatrix(const CorMatrix & value) = delete;
    CorMatrix(CorMatrix && value) = default;
    CorMatrix(vector<string> factors, arma::mat cor);
    ~CorMatrix() = default;

    pt::ptree to_ptree();
    static CorMatrix from_ptree(pt::ptree & value);

	vector<string> & get_factors();
	arma::mat get_cor();
	size_t n_factors();

    size_t pos_factor(string value);

    bool check_equation(Equation & value);
};

#endif // !FACTORCORRELATION_H__
