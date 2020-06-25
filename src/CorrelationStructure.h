#ifndef CORRELATIONSTRUCTURE_H__
#define CORRELATIONSTRUCTURE_H__

#include <vector>
#include <string>
#include "Equation.h"
#include "FactorCorrelation.h"

using std::vector;

class CorrelationStructure : public vector<Equation>
{
private:
    CorMatrix factorMatrix;
public:
	CorrelationStructure() = delete;
    CorrelationStructure(CorMatrix & fm);
	CorrelationStructure(const CorrelationStructure & value) = delete;
    CorrelationStructure(CorrelationStructure && value) = default;
	~CorrelationStructure() = default;

    static CorrelationStructure from_ptree(pt::ptree & value);
    pt::ptree to_ptree();

	void operator+(Equation & value);

	vector<string> & get_factors();
    void arrange();

	size_t n_factors();

    arma::mat get_sensitivities(const double * weights);
    arma::mat get_sensitivities(arma::vec weights);
    arma::mat get_sensitivities();

    arma::mat get_cor_factors();
    CorMatrix & get_factor_cor();
    size_t n_weights();

    void set_weights(arma::vec weights);
    arma::vec get_weights();
    vector<double> get_weights_v();

    arma::mat fitted_cor(const double * weights, double * R2);
    arma::mat fitted_cor(arma::vec weights, double * R2);
    arma::mat fitted_cor();

    double evaluate(const double * weights, CorMatrix & empiric);
    double evaluate(arma::vec weights, CorMatrix & empiric);
    double evaluate(CorMatrix & empiric);

    vector<double> lower_bounds();
    vector<double> upper_bounds();
};



#endif // !CORRELATIONSTRUCTURE_H__
