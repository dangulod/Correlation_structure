#include "Equation.h"
#include "FactorCorrelation.h"
#include "CorrelationStructure.h"
#include <nlopt.hpp>
#include <chrono>

void show_usage(string name)
{
    std::cerr << "Usage: " << name << " <options> parameters\n"
                  << "Options:\n\n"
                  << "\t-h, --help\t\t\tShow this help message\n"
                  << "\t-w, --weights file.json\t\tSpecify the path of the file with the sensitivities and the factor correlation\n"
                  << "\t-e, --empirical file.json\tSpecify the path of the file with the empirical correlation\n"
                  << "\t-o, --optimizer int\t\tSpecify the optimization algorithm\n\n"
                  << "\t\t1\tGN_ISRES\n"
                  << "\t\t2\tGN_ESCH\n"
                  << "\t\t3\tGN_MLSL\n"
                  << "\t\t4\tGN_CRS2_LM\n"
                  << "\t\t5\tLN_BOBYQA\n"
                  << "\t\t6\tLN_AUGLAG_EQ\n"
                  << "\t\t7\tLN_COBYLA\n\n"
                  << "\t-i,--iterations int\t\tSpecify the maximun number of iterations\n\n"
                  << "Example: " << name  << " -w weights.json -e empirical.json -o 1 -i 1000000"

                  << std::endl;
}

// ./EC_equation  -w /opt/share/data/EC21_AggEqu/EQ_CR_weights.json -e /opt/share/data/EC21_AggEqu/EQ_CR_cor.json -o 1 -i 10000

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        show_usage(argv[0]);
        return 1;
    }

    std::string file_weights, file_empirical;
    int optim, max_iter;
    nlopt::algorithm algorithm;

    for (int ii = 1; ii < argc; ii +=2)
    {
        std::string arg = argv[ii];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            return 1;
        } else if ((arg == "-w") || (arg == "--weights"))
        {
            file_weights = argv[ii + 1];
        } else if ((arg == "-e") || (arg == "--empirical"))
        {
            file_empirical = argv[ii + 1];
        } else if ((arg == "-o") || (arg == "--optimizer"))
        {
            optim = atoi(argv[ii + 1]);

            if ((optim < 1) | (optim > 7))
            {
                printf("-o, --optimizer must be between 1 and 7\n");
                return 2;
            }

            switch (optim)
            {
            case 1:
                algorithm = nlopt::GN_ISRES;
                break;
            case 2:
                algorithm = nlopt::GN_ESCH;
                break;
            case 3:
                algorithm = nlopt::GN_MLSL;
                break;
            case 4:
                algorithm = nlopt::GN_CRS2_LM;
                break;
            case 5:
                algorithm = nlopt::LN_BOBYQA;
                break;
            case 6:
                algorithm = nlopt::LN_AUGLAG_EQ;
                break;
            case 7:
                algorithm = nlopt::LN_COBYLA;
                break;
            default:
                break;
            }

        } else if ((arg == "-i") || (arg == "--iterations"))
        {
            max_iter = atoi(argv[ii + 1]);
        }
    }

    const auto startTime = std::chrono::high_resolution_clock::now();

    pt::ptree root;
    pt::read_json(file_weights, root);
    CorrelationStructure CS = CorrelationStructure::from_ptree(root);

    pt::read_json(file_empirical, root);
    CorMatrix M = CorMatrix::from_ptree(root);


    printf("sumsq(E - CS) = %g\n", CS.evaluate(M)); // Tiene que estar para asegurarse de que tiene la misma dimension

    CS.minimize(M, algorithm, max_iter);

    const auto endTime = std::chrono::high_resolution_clock::now();

    printf("Time elapsed: %f seconds\n", std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(endTime - startTime).count() / 1000);

    pt::write_json(file_weights, CS.to_ptree());

    return 0;
}

