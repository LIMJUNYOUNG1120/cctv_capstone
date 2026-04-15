#pragma once
#include <vector>

class HungarianAlgorithm {
public:
    double solve(
        const std::vector<std::vector<double>>& costMatrix,
        std::vector<int>& assignment);

private:
    void assignmentOptimal(
        int* assignment,
        double* cost,
        const double* distMatrix,
        int nOfRows,
        int nOfColumns);

    void buildAssignmentVector(
        int* assignment,
        bool* starMatrix,
        int nOfRows,
        int nOfColumns);

    void computeAssignmentCost(
        int* assignment,
        double* cost,
        const double* distMatrix,
        int nOfRows);

    void step2a(
        int* assignment, double* distMatrix,
        bool* starMatrix, bool* newStarMatrix,
        bool* primeMatrix, bool* coveredColumns,
        bool* coveredRows, int nOfRows,
        int nOfColumns, int minDim);

    void step2b(
        int* assignment, double* distMatrix,
        bool* starMatrix, bool* newStarMatrix,
        bool* primeMatrix, bool* coveredColumns,
        bool* coveredRows, int nOfRows,
        int nOfColumns, int minDim);

    void step3(
        int* assignment, double* distMatrix,
        bool* starMatrix, bool* newStarMatrix,
        bool* primeMatrix, bool* coveredColumns,
        bool* coveredRows, int nOfRows,
        int nOfColumns, int minDim);

    void step4(
        int* assignment, double* distMatrix,
        bool* starMatrix, bool* newStarMatrix,
        bool* primeMatrix, bool* coveredColumns,
        bool* coveredRows, int nOfRows,
        int nOfColumns, int minDim,
        int row, int col);

    void step5(
        int* assignment, double* distMatrix,
        bool* starMatrix, bool* newStarMatrix,
        bool* primeMatrix, bool* coveredColumns,
        bool* coveredRows, int nOfRows,
        int nOfColumns, int minDim);
};