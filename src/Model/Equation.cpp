#include "Equation.h"

Equation::Equation(string name, vector<std::pair<string, double>> sensib):
    vector<std::pair<string, double>>(sensib), name(name)
{
    for (size_t ii = 0; ii < sensib.size(); ii++)
    {
        for (size_t jj = ii + 1; jj < sensib.size(); jj++)
        {
            if (sensib[ii].first == sensib[jj].first) throw std::invalid_argument("Sensivity duplicated");
        }
    }
}

Equation Equation::operator=(Equation value)
{
    this->name = value.name;

    for (size_t ii = 0; ii < value.size(); ii++)
    {
        this->push_back(value.at(ii));
    }

    return *this;
}

pt::ptree Equation::to_ptree()
{
    pt::ptree root;
    root.put("name", this->name);

    pt::ptree sensib;

    for (auto && ii : *this)
    {
        sensib.put(ii.first, ii.second);
    }

    root.add_child("sensitivities", sensib);

    return root;
}

Equation Equation::from_ptree(pt::ptree & value)
{
    vector<std::pair<string, double>> sensib;

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("sensitivities"))
    {
        sensib.push_back(std::make_pair(ii.first, ii.second.get_value<double>()));
    }

    return Equation(value.get_child("name").get_value<string>(), sensib);
}

double Equation::get_weight(string factor)
{
    for (auto & ii : *this)
    {
        if (ii.first == factor) return ii.second;
    }

	return 0;
}

double Equation::get_weight(size_t n)
{
    return (*this)[n].second;
}

string Equation::get_name()
{
    return this->name;
}

void Equation::set_weights(size_t pos, double value)
{
    this->at(pos).second = value;
}

void Equation::set_weights(arma::vec value)
{
    if (value.size() != this->size()) throw std::invalid_argument("New weights size do not match with equation size");

    for (size_t ii = 0; ii < this->size(); ii++)
    {
        set_weights(ii, value(ii));
    }
}

double Equation::R2(CorMatrix & cor)
{
    arma::vec w(cor.n_factors());

    for (size_t ii = 0; ii < cor.n_factors(); ii++)
    {
        w.at(ii) = this->get_weight(cor.get_factors()[ii]);
    }

    return arma::as_scalar(w.t() * cor.get_cor() * w);
}

