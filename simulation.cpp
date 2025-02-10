#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/random.hpp>
#include <ctime>
#include <cstdint>

#include <fstream>

/***************** GRAPH ******************/
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/connected_components.hpp>
#include <iostream>

typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS> MyGraph;
typedef boost::graph_traits<MyGraph>::vertex_descriptor MyVertex;

/*************** END GRAPH ****************/

/*************** FINAL PHASE **************/

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/graph/random.hpp>
#include <boost/random/linear_congruential.hpp>

typedef boost::uniform_int<> UniformIntDistr;
typedef boost::variate_generator<boost::mt19937 &, UniformIntDistr> IntRNG;

/************ END FINAL PAHSE **************/

class MyVisitor : public boost::default_dfs_visitor
{
public:
    MyVisitor() : vv(new std::vector<MyVertex>()) {}

    void discover_vertex(MyVertex v, const MyGraph &g)
    { // note the lack of const
        if (boost::in_degree(v, g) != 0)
        { // only print the vertices in the connected component (I already did MCC and removed edges so all the extra vertices are isolated)
            std::cerr << v << std::endl;
            vv->push_back(v);
        }
        return;
    }
    std::vector<MyVertex> &GetVector() const { return *vv; }

private:
    boost::shared_ptr<std::vector<MyVertex>> vv;
};
/******************************************/

enum FinalState
{
    real_consensus,
    consensus_and_split,
    real_split,
    unfinished,
    FAIL
};

std::string to_string(FinalState final_state)
{
    switch (final_state)
    {
    case real_consensus:
        return "real_consensus";
    case consensus_and_split:
        return "consensus_and_split";
    case real_split:
        return "real_split";
    case unfinished:
        return "unfinished";
    case FAIL:
        return "FAIL";
    default:
        break;
    }
    return "None";
};

typedef struct SimulationParams
{

    int n_voters;
    float voter_prob_q;

    std::time_t seed;

    FinalState final_state;
    unsigned int n_vot_moves;
    unsigned int n_del_moves;
    unsigned int n_final_opinion_one;
    unsigned int n_connected_components;

    unsigned int minimal_degree;
    unsigned int maximal_degree;
    unsigned int average_degree;
    unsigned int median_degree;

    bool disagreeing_components_exist;

    std::vector<int> component_numbers; // assigns every individuals its component. Components are numbers starting with zero    
    std::vector<bool> opinions;         // assigns every individu
} SimulationParams;

typedef struct GraphStatistics
{
    unsigned int n_disagreeing_edges;
    unsigned int minimal_degree;
    unsigned int maximal_degree;
} GraphStatistics;

MyGraph construct_BGL_graph_from_adj_matrix(boost::numeric::ublas::matrix<bool> B, std::vector<bool> X, int n_voters)
{
    MyGraph g = MyGraph();

    int num_dis_edges = 0;

    for (int i = 0; i < n_voters; i++)
    {
        boost::add_vertex(g);
    }


    for (int i = 0; i < n_voters; i++)
    {

        for (int j = i + 1; j < n_voters; j++)
        {
            if (B(i, j) == 1)
            {
                boost::add_edge(i, j, g);
                if (X[i] != X[j])
                {
                    num_dis_edges = num_dis_edges + 1;
                }
            }
        }
    }

    // std::cout << "NUM DIS EDGES: " << num_dis_edges << "\n";
    return g;
}

std::vector<bool> initialize_opinion_vector(int n_voters)
{
    std::vector<bool> X(floor(n_voters / 2), true);
    std::vector<bool> X_2(floor(n_voters / 2), false);
    X.insert(X.end(), X_2.begin(), X_2.end());
    return X;
}

GraphStatistics compute_graph_statistics(boost::numeric::ublas::matrix<bool> B, std::vector<bool> X, int n_voters)
{

    GraphStatistics gs;

    gs.n_disagreeing_edges = 0;
    gs.minimal_degree = n_voters - 1;
    gs.maximal_degree = 0;

    for (int i = 0; i < n_voters; i++)
    {
        int degree_current_node = 0;

        for (int j = 0; j < n_voters; j++)
        {
            if (B(i, j) == true)
            {

                degree_current_node = degree_current_node + 1;

                if (i < j and X[i] != X[j])
                {
                    gs.n_disagreeing_edges++;
                }
            }
        }

        if (degree_current_node < gs.minimal_degree)
        {
            gs.minimal_degree = degree_current_node;
        }

        if (degree_current_node > gs.maximal_degree)
        {
            gs.maximal_degree = degree_current_node;
        }
    }
    return gs;
}

bool check_disagreeing_components(int num_components, std::vector<int> components, std::vector<bool> opinions)
{
    bool component_opinion_0_exists = false;
    bool component_opinion_1_exists = false;

    for (int i = 0; i < num_components; i++)
    {
        int component_opinion = -1;
        for (int j = 0; j < opinions.size(); j++)
        {
            if (component_opinion == -1 and components[j] == i)
            {
                component_opinion = opinions[j];
            }
            else if (components[j] == i and component_opinion != opinions[j])
            {
                break;
            }

            if (j == opinions.size() - 1 and component_opinion == 0)
            {
                component_opinion_0_exists = true;
            }
            if (j == opinions.size() - 1 and component_opinion == 1)
            {
                component_opinion_1_exists = true;
            }
        }
    }

    return component_opinion_0_exists and component_opinion_1_exists;

    // returns true if there are two disjoint components with differing opinions
}

SimulationParams simulate(unsigned int n_voters, float voter_prob_q)
{

    SimulationParams result;
    result.n_voters = n_voters;
    result.voter_prob_q = voter_prob_q;

    boost::numeric::ublas::scalar_matrix<bool> A(n_voters, n_voters, true);
    boost::numeric::ublas::identity_matrix<bool> eye(n_voters);
    boost::numeric::ublas::matrix<bool> complete_graph(n_voters, n_voters);
    complete_graph = A - eye;

    std::time_t now = std::time(0);
    result.seed = now;

    boost::random::mt19937 gen{static_cast<std::uint32_t>(now)};
    boost::random::uniform_int_distribution<> random_node(0, n_voters - 1);
    boost::random::bernoulli_distribution<> random_event(voter_prob_q);
    boost::random::bernoulli_distribution<> coin_flip(0.5);

    boost::numeric::ublas::matrix<bool> B(n_voters, n_voters);
    B = complete_graph;

    std::vector<bool> X = initialize_opinion_vector(n_voters);
    int N = std::accumulate(X.begin(), X.end(), 0);

    unsigned int count_vot = 0;
    unsigned int count_del = 0;
    unsigned int count_not_disagreeing = 0;

    int consecutive_unsuccessful_samples = 0;

    while (N != 0 and N != n_voters and consecutive_unsuccessful_samples < 2000000)
    {

        int site1 = random_node(gen);
        int site2 = random_node(gen);

        if (B(site1, site2) == true and X[site1] != X[site2])
        {
            if (random_event(gen) == 1)
            {
                X[site2] = X[site1];
                int opinion_change = 2 * X[site1] - 1;
                N = N + opinion_change;
                count_vot = count_vot + 1;
            }
            else
            {
                B(site1, site2) = 0;
                B(site2, site1) = 0;
                count_del = count_del + 1;
            }
            consecutive_unsuccessful_samples = 0;
        }
        else
        {
            if (X[site1] == X[site2])
            {
                count_not_disagreeing = count_not_disagreeing + 1;
            }
            consecutive_unsuccessful_samples = consecutive_unsuccessful_samples + 1;
        }
    }

    MyGraph g = construct_BGL_graph_from_adj_matrix(B, X, n_voters);

    std::vector<int> component(boost::num_vertices(g));
    size_t num_components = boost::connected_components(g, &component[0]);

    result.component_numbers = component;
    result.opinions = X;

    int size_of_first_component = 0;

    std::cout << "number of vertices in boost graph: " << boost::num_vertices(g) << "\n";


    for (size_t i = 0; i < boost::num_vertices(g); ++i)
    {
        if (component[i] == 0)
        {
            size_of_first_component = size_of_first_component + 1;
        }
    }

    // std::cout << "number of connected components: " << num_components << "\n";
    // std::cout << "size of first component: " << size_of_first_component << "\n";

    GraphStatistics graph_statistics = compute_graph_statistics(B, X, n_voters);

    if (graph_statistics.n_disagreeing_edges == 0)
    {
        if (num_components == 1 and (N == 0 or N == n_voters))
        {
            result.final_state = real_consensus;
        }
        else if (num_components > 1 and (N == 0 or N == n_voters))
        {
            result.final_state = consensus_and_split;
        }
        else if (num_components > 1 and N > 0 and N < n_voters)
        {
            result.final_state = real_split;
        }
        else
        {
            result.final_state = FAIL;
        }

        result.n_connected_components = num_components;
        result.n_vot_moves = count_vot;
        result.n_del_moves = count_del;
        result.n_final_opinion_one = N;
        result.maximal_degree = graph_statistics.maximal_degree;
        result.minimal_degree = graph_statistics.minimal_degree;
    }
    else
    {
        result.final_state = unfinished;
    }

    result.disagreeing_components_exist = check_disagreeing_components(num_components, component, X);

    return result;
}

int main()
{

    std::string commit_id;
    std::string output_filename;
    std::cout << "Please enter current commit id: "
              << "\n";
    std::cin >> commit_id;

    std::cout << "Please enter output filename: "
              << "\n";
    std::cin >> output_filename;

    std::ofstream filestream;
    filestream.open(output_filename, std::ios::app);

    filestream << "commit " << commit_id << "\n";
    filestream << "seed,n_voters,q,final_state,disagreeing_components_exist,n_vot_moves,n_del_moves,final_opinion_one,n_conn_comp,minimal_degree,maximal_degree"
               << "\n";

    filestream.close();

    for (int n = 6; n < 7; n++)
    {
        float p = (float)n / 20;

        for (int i = 0; i < 1000; i++)
        {
            std::cout << "n, i: " << n << ", " << i << "\n";

            SimulationParams result = simulate(512, p);

            filestream.open(output_filename, std::ios::app);

            filestream << result.seed << ","
                       << result.n_voters << ","
                       << result.voter_prob_q << ","
                       << to_string(result.final_state) << ","
                       << result.disagreeing_components_exist << ","
                       << result.n_vot_moves << ","
                       << result.n_del_moves << ","
                       << result.n_final_opinion_one << ","
                       << result.n_connected_components << ","
                       << result.minimal_degree << ","
                       << result.maximal_degree << "\n";

            for (auto v : result.opinions)
            {
                filestream << v << ",";
            }
            
            filestream << "\n";

	    
    	    std::cout << "length of component vector" << result.component_numbers.size() << "\n";
            
	    for (auto v : result.component_numbers)
            {
                filestream << v << ",";
            }

            filestream << "\n";
            
            filestream.close();
        }
    }
}
