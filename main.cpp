#include "Equation.h"
#include "FactorCorrelation.h"
#include "CorrelationStructure.h"
#include <nlopt.hpp>
#include <chrono>

class CS_data
{
public:
    CorrelationStructure *CS;
    CorMatrix *EM;
    CS_data() =default;
    CS_data(CorrelationStructure & CS, CorMatrix & EM)
    {
        this->CS = &CS;
        this->EM = &EM;
    }
    ~CS_data() = default;
};

static int count = 0;

double myfunc2(unsigned n, const double *x, double *grad, void *my_func_data)
{
    ++count;

    CS_data * data = static_cast<CS_data*>(my_func_data);

    double f = data->CS->evaluate(x, *data->EM);

    printf("iter: %i f: %f\r", count, f);

    return f;
}

int main()
{
    const auto startTime = std::chrono::high_resolution_clock::now();

    pt::ptree root;
    pt::read_json("/opt/share/data/EC21_AggEqu/13factores/equations.json", root);
    CorrelationStructure CS = CorrelationStructure::from_ptree(root);

    pt::read_json("/opt/share/data/EC21_AggEqu/13factores/total_matrix.json", root);
    CorMatrix M = CorMatrix::from_ptree(root);

    CS_data my_dat(CS, M);

    printf("sumsq(E - CS) = %g\n", CS.evaluate(M)); // Tiene que estar para asegurarse de que tiene la misma dimension

    //nlopt::opt optimizer(nlopt::GN_ISRES, CS.n_weights()); // GN_ISRES GN_ESCH GN_MLSL GN_CRS2_LM NLOPT_LN_BOBYQA
    nlopt::opt optimizer(nlopt::LN_AUGLAG_EQ, CS.n_weights());
    //nlopt::opt optimizer(nlopt::LN_COBYLA, CS.n_weights());


    optimizer.set_lower_bounds(CS.lower_bounds());
    optimizer.set_upper_bounds(CS.upper_bounds());

    std::vector<double> x0 = CS.get_weights_v();

    optimizer.set_min_objective(myfunc2, (void*)&my_dat);

    optimizer.set_xtol_rel(1e-9);
    optimizer.set_maxeval(2e4);
    //optimizer.set_population(1e3);

    double minf;

    try
    {
        nlopt::result result = optimizer.optimize(x0, minf);
        std::cout << "found minimum at f(" << ") = "
                  << std::setprecision(10) << minf << std::endl;
    }
    catch(std::exception &e)
    {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }

    CS.set_weights(x0);
    CS.get_sensitivities().print();

    printf("found minimum after %d evaluations\n", count);

    const auto endTime = std::chrono::high_resolution_clock::now();

    printf("Time elapsed: %f seconds\n", std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(endTime - startTime).count() / 1000);

    for (auto & ii: CS)
    {
        std::cout << "R2: " << ii.R2(CS.get_factor_cor()) << std::endl;
    }

    pt::write_json("/opt/share/data/EC21_AggEqu/13factores/equations.json", CS.to_ptree());

    return 0;
}

