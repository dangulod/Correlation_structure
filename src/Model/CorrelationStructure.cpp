#include "CorrelationStructure.h"

CorrelationStructure::CorrelationStructure(CorMatrix & fm) : factorMatrix(std::move(fm))
{

}

CorrelationStructure CorrelationStructure::from_ptree(pt::ptree & value)
{
    CorMatrix FC = CorMatrix::from_ptree(value.get_child("FactorCorrelation"));

    CorrelationStructure CS(FC);

    BOOST_FOREACH(const pt::ptree::value_type & ii, value.get_child("Equations"))
    {
        pt::ptree p = ii.second;
        Equation eq = Equation::from_ptree(p);

        CS + eq;
    }

    return CS;
}

pt::ptree CorrelationStructure::to_ptree()
{
    pt::ptree root;

    root.add_child("FactorCorrelation", this->factorMatrix.to_ptree());

    pt::ptree equations;

    for (auto && ii: *this)
    {
        equations.push_back(std::make_pair("", ii.to_ptree()));
    }

    root.add_child("Equations", equations);

    return root;
}

void CorrelationStructure::operator+(Equation & value)
{
    if (!this->factorMatrix.check_equation(value))
    {
		throw std::invalid_argument("Equation factors do not match with the factors of the correlation matrix");
	}

    for (auto & ii : *this)
    {
        if (value.get_name() == ii.get_name()) throw std::invalid_argument("Equation name duplicated");
    }

	this->push_back(std::move(value));
}

size_t CorrelationStructure::n_factors()
{
	return this->factorMatrix.n_factors();
}

size_t CorrelationStructure::n_weights()
{
    size_t n(0);

    for (auto & ii: *this)
    {
        for (auto & jj: ii)
        {
            if (jj.optim) ++n;
        }
    }
    return n;
}

void CorrelationStructure::arrange()
{
    std::sort(this->begin(), this->end(), [] (Equation a, Equation b) { return a.get_name() < b.get_name(); });
}

vector<string> & CorrelationStructure::get_factors()
{
	return this->factorMatrix.get_factors();
}

arma::mat CorrelationStructure::get_cor_factors()
{
    return this->factorMatrix.get_cor();
}

CorMatrix & CorrelationStructure::get_factor_cor()
{
    return this->factorMatrix;
}

arma::mat CorrelationStructure::get_sensitivities()
{
	arma::mat cc(this->size(), this->n_factors(), arma::fill::zeros);

	for (size_t ii = 0, ne = this->size(); ii < ne; ii++)
	{
		for (size_t jj = 0, nf = this->n_factors(); jj < nf; jj++)
		{
			cc(ii, jj) = (*this)[ii].get_weight(this->get_factors()[jj]);
		}
	}
	return cc;
}

arma::mat CorrelationStructure::get_sensitivities(arma::vec weigths)
{
    arma::mat cc(this->size(), this->n_factors(), arma::fill::zeros);

    size_t ww(0);

    for (size_t ii = 0; ii < this->size(); ii++)
    {
        for (size_t jj = 0; jj < (*this)[ii].size(); jj++)
        {
            if ((*this)[ii][jj].optim)
            {
                cc(ii, this->factorMatrix.pos_factor((*this)[ii][jj].name)) = weigths(ww);
                ww++;
            } else
            {
                cc(ii, this->factorMatrix.pos_factor((*this)[ii][jj].name)) = (*this)[ii][jj].weight;
            }
        }
    }
    return cc;
}

arma::mat CorrelationStructure::get_sensitivities(const double * weigths)
{
    arma::mat cc(this->size(), this->n_factors(), arma::fill::zeros);

    size_t ww(0);

    for (size_t ii = 0; ii < this->size(); ii++)
    {
        for (size_t jj = 0; jj < (*this)[ii].size(); jj++)
        {
            if ((*this)[ii][jj].optim)
            {
                cc(ii, this->factorMatrix.pos_factor((*this)[ii][jj].name)) = weigths[ww];
                ww++;
            }
            else
            {
                cc(ii, this->factorMatrix.pos_factor((*this)[ii][jj].name)) = (*this)[ii][jj].weight;
            }
        }
    }
    return cc;
}


void CorrelationStructure::set_weights(arma::vec weights)
{
    if (this->n_weights() != weights.size()) throw std::invalid_argument("Vectors do not have the same size");

    size_t ss(0);

    for (auto & jj: *this)
    {
        for (size_t rr = 0; rr < jj.size(); rr++)
        {
            if (jj.at(rr).optim)
            {
                jj.set_weights(rr, weights(ss));
                ss++;
            }
        }
    }
}

arma::vec CorrelationStructure::get_weights()
{
    arma::vec w(this->n_weights());

    size_t ss(0);

    for (auto & jj: *this)
    {
        for (size_t rr = 0; rr < jj.size(); rr++)
        {
            if (jj.at(rr).optim)
            {
                w(ss) = jj.get_weight(rr);
                ss++;
            }
        }
    }

    return w;
}

vector<double> CorrelationStructure::get_weights_v()
{
    vector<double> w(this->n_weights());

    size_t ss(0);

    for (auto & jj: *this)
    {
        for (size_t rr = 0; rr < jj.size(); rr++)
        {
            if (jj.at(rr).optim)
            {
                w[ss] = jj.get_weight(rr);
                ss++;
            }
        }
    }

    return w;
}

arma::mat CorrelationStructure::fitted_cor()
{
	arma::mat cc = get_sensitivities();
	
    arma::mat cor =  cc * factorMatrix.get_cor() * cc.t();
    cor.diag().ones();

    return cor;
}

arma::mat CorrelationStructure::fitted_cor(arma::vec weights, double * R2)
{
    arma::mat cc = get_sensitivities(weights);
    arma::mat cor =  cc * factorMatrix.get_cor() * cc.t();

    arma::vec d = cor.diag();
    *R2 = arma::accu(d(arma::find(d > 1)));

    cor.diag().ones();

    return cor;
}

arma::mat CorrelationStructure::fitted_cor(const double * weights, double * R2)
{
    arma::mat cc = get_sensitivities(weights);
    arma::mat cor =  cc * factorMatrix.get_cor() * cc.t();

    arma::vec d = cor.diag();
    *R2 = arma::accu(d(arma::find(d > 1)));

    cor.diag().ones();

    return cor;
}

double CorrelationStructure::evaluate(const double * weights, CorMatrix & empiric)
{
    double R2 = 0;
    arma::mat fit = this->fitted_cor(weights, &R2);
    double r = arma::accu(pow(empiric.get_cor() - fit, 2));

    return r + r * R2;
}

double CorrelationStructure::evaluate(arma::vec weights, CorMatrix & empiric)
{
    double R2 = 0;
    arma::mat fit = this->fitted_cor(weights, &R2);
    double r = arma::accu(pow(empiric.get_cor() - fit, 2));

    return r + r * R2;
}

double CorrelationStructure::evaluate(CorMatrix & empiric)
{
    if (empiric.n_factors() != this->size()) throw std::invalid_argument("Matrix does not have the right dimension");

    for (size_t ii = 0; ii < this->size(); ii++)
    {
        if (empiric.get_factors()[ii] != (*this)[ii].get_name()) throw std::invalid_argument("Equations do not match or the order is not correct");
    }

    return arma::accu(pow(empiric.get_cor() - this->fitted_cor(), 2));
}

vector<double> CorrelationStructure::lower_bounds()
{
    vector<double> lower(this->n_weights());
    std::fill(lower.begin(), lower.end(), -1);

    return lower;
}

vector<double> CorrelationStructure::upper_bounds()
{
    vector<double> upper(this->n_weights());
    std::fill(upper.begin(), upper.end(), 1);

    return upper;
}

double CorrelationStructure::minimize(CorMatrix & empirical, nlopt::algorithm algorithm, int max_iter)
{
    struct Arg_data
    {
        int iter;
        CorrelationStructure *CS;
        CorMatrix *EM;
        Arg_data(CorrelationStructure & CS, CorMatrix & EM) : iter(0)
        {
            this->CS = &CS;
            this->EM = &EM;
        }
        ~Arg_data() = default;
    };

    auto fitnes = [] (const std::vector<double> &x, std::vector<double> &grad, void * args)
    {
        Q_UNUSED(grad);
        Q_UNUSED(args);

        Arg_data * data = static_cast<Arg_data*>(args);

        double f = data->CS->evaluate(x, *data->EM);

        printf("iter: %i f: %f\r", data->iter, f);

        data->iter++;
        return f;
    };

    // GN_ISRES GN_ESCH GN_MLSL GN_CRS2_LM LN_BOBYQA LN_AUGLAG_EQ LN_COBYLA
    nlopt::opt optimizer(algorithm, this->n_weights());

    optimizer.set_lower_bounds(this->lower_bounds());
    optimizer.set_upper_bounds(this->upper_bounds());

    std::vector<double> x0 = this->get_weights_v();

    Arg_data arg(*this, empirical);

    optimizer.set_min_objective(fitnes, (void*)&arg);

    optimizer.set_xtol_rel(1e-9);
    optimizer.set_maxeval(max_iter);

    double minf;

    try
    {
        optimizer.optimize(x0, minf);
        std::cout << "found minimum at = "
                  << std::setprecision(10) << minf << std::endl;
    }
    catch(std::exception &e)
    {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }

    printf("found minimum after %d evaluations\n", arg.iter);

    this->set_weights(x0);

    return 0;

}





