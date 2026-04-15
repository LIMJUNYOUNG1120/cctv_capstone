#include "hungarian.h"
#include <cmath>
#include <limits>
#include <cstring>

double HungarianAlgorithm::solve(
    const std::vector<std::vector<double>>& costMatrix,
    std::vector<int>& assignment) {

    int nOfRows = costMatrix.size();
    int nOfColumns = costMatrix[0].size();
    int minDim = std::min(nOfRows, nOfColumns);

    double cost = 0.0;
    int* assignmentArr = new int[nOfRows];
    double* distMatrix = new double[nOfRows * nOfColumns];

    for (int r = 0; r < nOfRows; r++)
        for (int c = 0; c < nOfColumns; c++)
            distMatrix[r + c * nOfRows] = costMatrix[r][c];

    assignmentOptimal(assignmentArr, &cost, distMatrix, nOfRows, nOfColumns);

    assignment.clear();
    for (int r = 0; r < nOfRows; r++)
        assignment.push_back(assignmentArr[r]);

    delete[] assignmentArr;
    delete[] distMatrix;
    return cost;
}

void HungarianAlgorithm::assignmentOptimal(
    int* assignment, double* cost,
    const double* distMatrix,
    int nOfRows, int nOfColumns) {

    int minDim = std::min(nOfRows, nOfColumns);
    bool* starMatrix = new bool[nOfRows * nOfColumns]();
    bool* primeMatrix = new bool[nOfRows * nOfColumns]();
    bool* newStarMatrix = new bool[nOfRows * nOfColumns]();
    bool* coveredColumns = new bool[nOfColumns]();
    bool* coveredRows = new bool[nOfRows]();

    double* distMatrixTemp = new double[nOfRows * nOfColumns];
    std::memcpy(distMatrixTemp, distMatrix,
        sizeof(double) * nOfRows * nOfColumns);

    *cost = 0;
    std::fill(assignment, assignment + nOfRows, -1);

    for (int r = 0; r < nOfRows; r++) {
        double minVal = std::numeric_limits<double>::max();
        for (int c = 0; c < nOfColumns; c++)
            minVal = std::min(minVal, distMatrixTemp[r + c * nOfRows]);
        for (int c = 0; c < nOfColumns; c++)
            distMatrixTemp[r + c * nOfRows] -= minVal;
    }

    for (int r = 0; r < nOfRows; r++) {
        for (int c = 0; c < nOfColumns; c++) {
            if (distMatrixTemp[r + c * nOfRows] == 0 &&
                !coveredColumns[c] && !coveredRows[r]) {
                starMatrix[r + c * nOfRows] = true;
                coveredColumns[c] = true;
                coveredRows[r] = true;
                break;
            }
        }
    }

    std::fill(coveredRows, coveredRows + nOfRows, false);

    step2b(assignment, distMatrixTemp, starMatrix, newStarMatrix,
        primeMatrix, coveredColumns, coveredRows,
        nOfRows, nOfColumns, minDim);

    buildAssignmentVector(assignment, starMatrix, nOfRows, nOfColumns);
    computeAssignmentCost(assignment, cost, distMatrix, nOfRows);

    delete[] starMatrix;
    delete[] primeMatrix;
    delete[] newStarMatrix;
    delete[] coveredColumns;
    delete[] coveredRows;
    delete[] distMatrixTemp;
}

void HungarianAlgorithm::buildAssignmentVector(
    int* assignment, bool* starMatrix,
    int nOfRows, int nOfColumns) {
    for (int r = 0; r < nOfRows; r++)
        for (int c = 0; c < nOfColumns; c++)
            if (starMatrix[r + c * nOfRows]) {
                assignment[r] = c;
                break;
            }
}

void HungarianAlgorithm::computeAssignmentCost(
    int* assignment, double* cost,
    const double* distMatrix, int nOfRows) {
    for (int r = 0; r < nOfRows; r++)
        if (assignment[r] >= 0)
            *cost += distMatrix[r + assignment[r] * nOfRows];
}

void HungarianAlgorithm::step2a(
    int* assignment, double* distMatrix,
    bool* starMatrix, bool* newStarMatrix,
    bool* primeMatrix, bool* coveredColumns,
    bool* coveredRows, int nOfRows,
    int nOfColumns, int minDim) {

    for (int c = 0; c < nOfColumns; c++)
        for (int r = 0; r < nOfRows; r++)
            if (starMatrix[r + c * nOfRows])
                coveredColumns[c] = true;

    step2b(assignment, distMatrix, starMatrix, newStarMatrix,
        primeMatrix, coveredColumns, coveredRows,
        nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step2b(
    int* assignment, double* distMatrix,
    bool* starMatrix, bool* newStarMatrix,
    bool* primeMatrix, bool* coveredColumns,
    bool* coveredRows, int nOfRows,
    int nOfColumns, int minDim) {

    int count = 0;
    for (int c = 0; c < nOfColumns; c++)
        if (coveredColumns[c]) count++;

    if (count == minDim)
        buildAssignmentVector(assignment, starMatrix, nOfRows, nOfColumns);
    else
        step3(assignment, distMatrix, starMatrix, newStarMatrix,
            primeMatrix, coveredColumns, coveredRows,
            nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step3(
    int* assignment, double* distMatrix,
    bool* starMatrix, bool* newStarMatrix,
    bool* primeMatrix, bool* coveredColumns,
    bool* coveredRows, int nOfRows,
    int nOfColumns, int minDim) {

    bool zerosFound = true;
    while (zerosFound) {
        zerosFound = false;
        for (int c = 0; c < nOfColumns; c++) {
            if (coveredColumns[c]) continue;
            for (int r = 0; r < nOfRows; r++) {
                if (coveredRows[r]) continue;
                if (distMatrix[r + c * nOfRows] != 0) continue;
                primeMatrix[r + c * nOfRows] = true;
                int starCol = -1;
                for (int c2 = 0; c2 < nOfColumns; c2++)
                    if (starMatrix[r + c2 * nOfRows]) {
                        starCol = c2; break;
                    }
                if (starCol == -1) {
                    step4(assignment, distMatrix, starMatrix,
                        newStarMatrix, primeMatrix, coveredColumns,
                        coveredRows, nOfRows, nOfColumns, minDim, r, c);
                    return;
                }
                else {
                    coveredRows[r] = true;
                    coveredColumns[starCol] = false;
                    zerosFound = true;
                }
            }
        }
    }
    step5(assignment, distMatrix, starMatrix, newStarMatrix,
        primeMatrix, coveredColumns, coveredRows,
        nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step4(
    int* assignment, double* distMatrix,
    bool* starMatrix, bool* newStarMatrix,
    bool* primeMatrix, bool* coveredColumns,
    bool* coveredRows, int nOfRows,
    int nOfColumns, int minDim,
    int row, int col) {

    std::memcpy(newStarMatrix, starMatrix,
        sizeof(bool) * nOfRows * nOfColumns);
    newStarMatrix[row + col * nOfRows] = true;

    int starCol = col, starRow = -1;
    for (int r = 0; r < nOfRows; r++)
        if (starMatrix[r + starCol * nOfRows]) {
            starRow = r; break;
        }

    while (starRow >= 0) {
        newStarMatrix[starRow + starCol * nOfRows] = false;
        int primeRow = starRow, primeCol = -1;
        for (int c = 0; c < nOfColumns; c++)
            if (primeMatrix[primeRow + c * nOfRows]) {
                primeCol = c; break;
            }
        newStarMatrix[primeRow + primeCol * nOfRows] = true;
        starCol = primeCol;
        starRow = -1;
        for (int r = 0; r < nOfRows; r++)
            if (starMatrix[r + starCol * nOfRows]) {
                starRow = r; break;
            }
    }

    std::memcpy(starMatrix, newStarMatrix,
        sizeof(bool) * nOfRows * nOfColumns);
    std::fill(primeMatrix, primeMatrix + nOfRows * nOfColumns, false);
    std::fill(coveredRows, coveredRows + nOfRows, false);

    step2a(assignment, distMatrix, starMatrix, newStarMatrix,
        primeMatrix, coveredColumns, coveredRows,
        nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step5(
    int* assignment, double* distMatrix,
    bool* starMatrix, bool* newStarMatrix,
    bool* primeMatrix, bool* coveredColumns,
    bool* coveredRows, int nOfRows,
    int nOfColumns, int minDim) {

    double h = std::numeric_limits<double>::max();
    for (int r = 0; r < nOfRows; r++) {
        if (coveredRows[r]) continue;
        for (int c = 0; c < nOfColumns; c++) {
            if (coveredColumns[c]) continue;
            h = std::min(h, distMatrix[r + c * nOfRows]);
        }
    }

    for (int r = 0; r < nOfRows; r++)
        for (int c = 0; c < nOfColumns; c++) {
            if (coveredRows[r])
                distMatrix[r + c * nOfRows] += h;
            if (!coveredColumns[c])
                distMatrix[r + c * nOfRows] -= h;
        }

    step3(assignment, distMatrix, starMatrix, newStarMatrix,
        primeMatrix, coveredColumns, coveredRows,
        nOfRows, nOfColumns, minDim);
}