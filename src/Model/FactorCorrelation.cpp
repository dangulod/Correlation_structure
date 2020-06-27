#include "FactorCorrelation.h"


void isCor(arma::mat cor)
{
    if (cor.n_cols != cor.n_rows)
    {
        throw std::invalid_argument("Correlation matrix is not squared");
    }

    for (unsigned long i = 0,  l = cor.n_cols; i < l; i++)
    {
        for (unsigned long j = 0; j < l; j++)
        {
            if (i == j)
            {
                if ( fabs(cor(i, j) - 1.0 ) > EPSILON )
                {
                    throw std::invalid_argument("The diagonal of the correlation matrix must be 1");
                }
            }
            else
            {
                if ( fabs(cor(i, j) - cor(j, i)) > EPSILON )
                {
                    throw std::invalid_argument("Correlation matrix is not symetric");
                }
            }
        }
    }
}

CorMatrix::CorMatrix(vector<string> factors, arma::mat cor) : factors(factors)
{
    if (cor.n_cols != this->factors.size()) throw std::invalid_argument("Factors and correlation matrix do not have the same length");

    isCor(cor);

    this->cor = cor;
}


CorMatrix CorMatrix::from_ptree(pt::ptree & value)
{
    vector<string> factors;

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("factors"))
    {
        factors.push_back(ii.second.get_value<string>());
    }

    arma::mat cor(factors.size(), factors.size());

    size_t c_ii = 0;

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("cor"))
    {
        size_t c_jj = 0;
        BOOST_FOREACH(const pt::ptree::value_type & jj, ii.second.get_child(""))
        {
            cor.at(c_ii, c_jj) = jj.second.get_value<double>();
            c_jj++;
        }
        c_ii++;
    }
    return CorMatrix(factors, cor);
}

pt::ptree CorMatrix::to_ptree()
{
    pt::ptree root;

    pt::ptree factor_nodes;

    for (auto && ii: this->factors)
    {
        pt::ptree factor_no;
        factor_no.put("", ii);

        factor_nodes.push_back(std::make_pair("", factor_no));
    }

    root.add_child("factors", factor_nodes);

    pt::ptree matrix_node;

    for (size_t && ii = 0; ii < this->n_factors(); ii++)
    {
        pt::ptree row;

        for (size_t && jj = 0; jj < this->cor.n_cols; jj++)
        {
            pt::ptree cell;
            cell.put_value(this->cor(ii, jj));
            row.push_back(std::make_pair("", cell));
        }
        matrix_node.push_back(std::make_pair("", row));
    }
    root.add_child("cor", matrix_node);

    return root;
}

vector<string> & CorMatrix::get_factors()
{
	return this->factors;
}

arma::mat CorMatrix::get_cor()
{
    return this->cor;
}

bool CorMatrix::check_equation(Equation & value)
{
    size_t tt = 0;

    for (auto && ii: value )
    {
        tt += (std::find(this->factors.begin(), this->factors.end(), ii.name) != this->factors.end());
    }

    return tt == value.size() ? 1 : 0;
}

size_t CorMatrix::pos_factor(string value)
{
    for (size_t ii = 0; ii < this->n_factors(); ii++)
    {
        if (this->factors[ii] == value) return ii;
    }

    return this->n_factors();
}


size_t CorMatrix::n_factors()
{
	return this->factors.size();
}
