/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
* 
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
* 
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
* 
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                    
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cstdio>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Tile.h"
#include "Chip.h"
#include "ProcessingUnit.h"
#include "SubArray.h"
#include "Definition.h"
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <ctime>

using namespace std;

vector<vector<double> > getNetStructure(const string &inputfile);

int main(int argc, char * argv[]) {   

    auto start = chrono::high_resolution_clock::now();
    
    gen.seed(0);
    
    vector<vector<double> > netStructure;
    netStructure = getNetStructure(argv[1]);
    
    // define weight/input/memory precision from wrapper
    param->synapseBit = atoi(argv[2]);              // precision of synapse weight
    param->numBitInput = atoi(argv[3]);             // precision of input neural activation
    param->numRowSubArray = atoi(argv[4]);             // number row of subarray
    param->numRowParallel = atoi(argv[5]);             // number of enabled rows of subarray (partial paralle mode)


    if (param->cellBit > param->synapseBit) {
        cout << "ERROR!: Memory precision is even higher than synapse precision, please modify 'cellBit' in Param.cpp!" << endl;
        param->cellBit = param->synapseBit;
    }

    // 1.4 update: warning for the incompatible modes for a given device technology

    if ( param->memcelltype == 2 && param->technode <=14) {

        cout << "ERROR!: RRAM-CIM is not supported beyond 22 nm!" << endl;
        exit(-1);
    }

    if ( param->deviceroadmap == 1 && param->technode <=14) {

        cout << "ERROR!: HP technology is not supported for 14 nm and beyond!" << endl;
        exit(-1);
    }

    if ( param->temp != 300 && param->technode <=14) {

        cout << "ERROR!: only 300K operation is supported for 14 nm and beyond!" << endl;
        exit(-1);
    }

    if ((param->sync_data_transfer)) {
    if (!((param->sync_data_transfer) && (param->globalBusType == true) && (param->chipActivation == true) 
    && (param->novelMapping == true) && (param->pipeline== true) 
    && (param->synchronous == true) && (param->globalBufferType == false)))

    {   cout << "ERROR!: sync_data_transfer mode not supported for the current parameter combimation!" << endl;
        exit(-1);
    }
    }

    /*** initialize operationMode as default ***/
    param->conventionalParallel = 0;
    param->conventionalSequential = 0;
    param->BNNparallelMode = 0;                // parallel BNN
    param->BNNsequentialMode = 0;              // sequential BNN
    param->XNORsequentialMode = 0;           // Use several multi-bit RRAM as one synapse
    param->XNORparallelMode = 0;         // Use several multi-bit RRAM as one synapse
    switch(param->operationmode) {
        case 6:     param->XNORparallelMode = 1;               break;     
        case 5:     param->XNORsequentialMode = 1;             break;     
        case 4:     param->BNNparallelMode = 1;                break;     
        case 3:     param->BNNsequentialMode = 1;              break;    
        case 2:     param->conventionalParallel = 1;           break;     
        case 1:     param->conventionalSequential = 1;         break;    
        case -1:    break;
        default:    exit(-1);
    }
    
    if (param->XNORparallelMode || param->XNORsequentialMode) {
        param->numRowPerSynapse = 2;
    } else {
        param->numRowPerSynapse = 1;
    }
    if (param->BNNparallelMode) {
        param->numColPerSynapse = 2;
    } else if (param->XNORparallelMode || param->XNORsequentialMode || param->BNNsequentialMode) {
        param->numColPerSynapse = 1;
    } else {
        param->numColPerSynapse = ceil((double)param->synapseBit/(double)param->cellBit); 
    }
    

    // 1.4 update : Implementation for conventional sequential 
    if ( ( param->conventionalSequential == 1 || param->conventionalSequential == 3 || param->conventionalSequential == 5 ) && (param->memcelltype==1) )
    {
    param->numColMuxed = param->numColPerSynapse;
    }

    double maxPESizeNM, maxTileSizeCM, numPENM;
    vector<int> markNM;
    vector<int> pipelineSpeedUp;
    markNM = ChipDesignInitialize(inputParameter, tech, cell, false, netStructure, &maxPESizeNM, &maxTileSizeCM, &numPENM);
    pipelineSpeedUp = ChipDesignInitialize(inputParameter, tech, cell, true, netStructure, &maxPESizeNM, &maxTileSizeCM, &numPENM);
    
    double desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM;
    int numTileRow, numTileCol;
    
    vector<vector<double> > numTileEachLayer;
    vector<vector<double> > utilizationEachLayer;
    vector<vector<double> > speedUpEachLayer;
    vector<vector<double> > tileLocaEachLayer;
    
    vector<vector<double>> peDup, subArrayDup;

    numTileEachLayer = ChipFloorPlan(true, false, false, netStructure, markNM, 
                    maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
                    &desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol, &peDup, &subArrayDup);    
    

    utilizationEachLayer = ChipFloorPlan(false, true, false, netStructure, markNM, 
                    maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
                    &desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol, nullptr, nullptr);
    
    speedUpEachLayer = ChipFloorPlan(false, false, true, netStructure, markNM,
                    maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
                    &desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol, nullptr, nullptr);
                    
    tileLocaEachLayer = ChipFloorPlan(false, false, false, netStructure, markNM,
                    maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
                    &desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol, nullptr, nullptr);
    
    // cout << "------------------------------ FloorPlan --------------------------------" <<  endl;
    // cout << endl;
    // cout << "Tile and PE size are optimized to maximize memory utilization ( = memory mapped by synapse / total memory on chip)" << endl;
    // cout << endl;
    // if (!param->novelMapping) {
    //     cout << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
    //     cout << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
    // } else {
    //     cout << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
    //     cout << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
    //     cout << "Desired Novel Mapped Tile Storage Size: " << numPENM << "x" << desiredPESizeNM << "x" << desiredPESizeNM << endl;
    // }
    // cout << "User-defined SubArray Size: " << param->numRowSubArray << "x" << param->numColSubArray << endl;
    // cout << endl;
    // cout << "----------------- # of tile used for each layer -----------------" <<  endl;
    // double totalNumTile = 0;
    // for (int i=0; i<netStructure.size(); i++) {
    //     cout << "layer" << i+1 << ": " << numTileEachLayer[0][i] * numTileEachLayer[1][i] << endl;
    //     totalNumTile += numTileEachLayer[0][i] * numTileEachLayer[1][i];
    // }
    // cout << endl;

    // cout << "----------------- Speed-up of each layer ------------------" <<  endl;
    // for (int i=0; i<netStructure.size(); i++) {
    //     cout << "layer" << i+1 << ": " << speedUpEachLayer[0][i] * speedUpEachLayer[1][i] << endl;
    // }
    // cout << endl;
    
    // cout << "----------------- Utilization of each layer ------------------" <<  endl;
    // double realMappedMemory = 0;
    // for (int i=0; i<netStructure.size(); i++) {
    //     cout << "layer" << i+1 << ": " << utilizationEachLayer[i][0] << endl;
    //     realMappedMemory += numTileEachLayer[0][i] * numTileEachLayer[1][i] * utilizationEachLayer[i][0];
    // }
    // cout << "Memory Utilization of Whole Chip: " << realMappedMemory/totalNumTile*100 << " % " << endl;
    // cout << endl;
    // cout << "---------------------------- FloorPlan Done ------------------------------" <<  endl;
    // cout << endl;
    // cout << endl;
    // cout << endl;
    


    // cout << "----------------- Layer-wise Architecture Configuration ------------------" << endl;
    // for (int i = 0; i < netStructure.size(); i++) {
    //     cout << "Layer " << i + 1 << ":" << endl;
    //     cout << "  Mapping Type: " << (markNM[i] == 1 ? "Novel" : "Conventional") << endl;
    //     cout << "  Tile count (row x col): " << numTileEachLayer[0][i] << " x " << numTileEachLayer[1][i]
    //         << " = " << numTileEachLayer[0][i] * numTileEachLayer[1][i] << endl;

    //     if (markNM[i] == 1) {
    //         cout << "  PE size: " << desiredPESizeNM << " x " << desiredPESizeNM << endl;
    //         cout << "  PE count: " << numPENM << endl;
    //     } else {
    //         cout << "  Tile size: " << desiredTileSizeCM << " x " << desiredTileSizeCM << endl;
    //         cout << "  PE size: " << desiredPESizeCM << " x " << desiredPESizeCM << endl;
    //     }
    //     cout << endl;
    // }


    // for (int i = 0; i < netStructure.size(); i++) {
    //     cout << "Layer " << i+1 << " Duplication Info:\n";
    //     cout << "  PE Duplication(row): " << peDup[0][i] << "\n";
    //     cout << "  PE Duplication(col): " << peDup[1][i] << "\n";
    //     cout << "  SubArray Duplication(row): " << subArrayDup[0][i] << "\n";
    //     cout << "  SubArray Duplication(col): " << subArrayDup[1][i] << "\n";
    // }

    int bottleneckLayer = -1;              // 0-based
    double bottleneckReadLatency = 0.0;
    double bottleneckBufferLatency = 0.0;
    double bottleneckICLatency = 0.0;
    double bottleneckCoreLatencyADC = 0.0;
    double bottleneckCoreLatencyAccum = 0.0;
    double bottleneckCoreLatencyOther = 0.0;

    double numComputation = 0;
    for (int i=0; i<netStructure.size(); i++) {
        numComputation += 2*(netStructure[i][0] * netStructure[i][1] * netStructure[i][2] * netStructure[i][3] * netStructure[i][4] * netStructure[i][5]);
    }

    //numComputation *= 2; // 앙상블 모델은 이미지당 계산량 2배 

    
    ChipInitialize(inputParameter, tech, cell, netStructure, markNM, numTileEachLayer,
                    numPENM, desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM, numTileRow, numTileCol);
            
    double chipHeight, chipWidth, chipArea, chipAreaIC, chipAreaADC, chipAreaAccum, chipAreaOther, chipAreaArray;
    double CMTileheight = 0;
    double CMTilewidth = 0;
    double NMTileheight = 0;
    double NMTilewidth = 0;
    vector<double> chipAreaResults;
    
    chipAreaResults = ChipCalculateArea(inputParameter, tech, cell, desiredNumTileNM, numPENM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM, numTileRow, 
                    &chipHeight, &chipWidth, &CMTileheight, &CMTilewidth, &NMTileheight, &NMTilewidth);     
    
    chipArea = chipAreaResults[0];
    chipAreaIC = chipAreaResults[1];
    chipAreaADC = chipAreaResults[2];
    chipAreaAccum = chipAreaResults[3];
    chipAreaOther = chipAreaResults[4];
    chipAreaArray = chipAreaResults[5];

    double clkPeriod = 0;
    double layerclkPeriod = 0;
    
    double chipReadLatency = 0;
    double chipReadDynamicEnergy = 0;
    double chipLeakageEnergy = 0;
    double chipLeakage = 0;
    double chipbufferLatency = 0;
    double chipbufferReadDynamicEnergy = 0;
    double chipicLatency = 0;
    double chipicReadDynamicEnergy = 0;
    
    double chipLatencyADC = 0;
    double chipLatencyAccum = 0;
    double chipLatencyOther = 0;
    double chipEnergyADC = 0;
    double chipEnergyAccum = 0;
    double chipEnergyOther = 0;
    
    double layerReadLatency = 0;
    double layerReadDynamicEnergy = 0;
    double tileLeakage = 0;
    // Anni update: leakage of SRAM when partial subarray in use
    double tileLeakageSRAMInUse = 0;
    double layerbufferLatency = 0;
    double layerbufferDynamicEnergy = 0;
    double layericLatency = 0;
    double layericDynamicEnergy = 0;
    
    double coreLatencyADC = 0;
    double coreLatencyAccum = 0;
    double coreLatencyOther = 0;
    double coreEnergyADC = 0;
    double coreEnergyAccum = 0;
    double coreEnergyOther = 0;
    
    const int ENSEMBLE_MODELS = 1;

    // per-model로만 바꾼 speedUpEachLayer
    std::vector<std::vector<double>> speedUpEachLayer_part(2, std::vector<double>(netStructure.size(), 0.0));
    for (int l=0; l<netStructure.size(); ++l) {
        int dupPhysRow = peDup[0][l]*subArrayDup[0][l];
        int dupPhysCol = peDup[1][l]*subArrayDup[1][l];
        int denom_phys = dupPhysRow * dupPhysCol;

        int extraRow = (int)ceil((double)pipelineSpeedUp[l] / (denom_phys * ENSEMBLE_MODELS));

        speedUpEachLayer_part[0][l] = dupPhysRow * extraRow;
        speedUpEachLayer_part[1][l] = dupPhysCol;
    }


    if (param->synchronous){
        // calculate clkFreq
        for (int i=0; i<netStructure.size(); i++) {     
            // Anni update: add &tileLeakageSRAMInUse
            ChipCalculatePerformance(inputParameter, tech, cell, i, argv[2*i+6], argv[2*i+6], argv[2*i+7], netStructure[i][6],
                        netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer_part, tileLocaEachLayer,
                        numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth,
                        &layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &tileLeakageSRAMInUse, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
                        &coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, true, &layerclkPeriod);
            if(clkPeriod < layerclkPeriod){
                clkPeriod = layerclkPeriod;
            }           
        }       
        cout<<"clkPeriod: "<<clkPeriod<<endl;
        if(param->clkFreq > 1/clkPeriod){
            param->clkFreq = 1/clkPeriod;
        }
    }
    
    printf("tech.vdd : %f\n", tech.vdd);

    cout << "-------------------------------------- Hardware Performance --------------------------------------" <<  endl;  
    if (! param->pipeline) {
        // layer-by-layer process
        // show the detailed hardware performance for each layer
        for (int i=0; i<netStructure.size(); i++) {
            cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;
            // Anni update: add &tileLeakageSRAMInUse
            ChipCalculatePerformance(inputParameter, tech, cell, i, argv[2*i+6], argv[2*i+6], argv[2*i+7], netStructure[i][6],
                        netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer_part, tileLocaEachLayer,
                        numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth,
                        &layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &tileLeakageSRAMInUse, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
                        &coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, false, &layerclkPeriod);
            if (param->synchronous) {
                layerReadLatency *= clkPeriod;
                layerbufferLatency *= clkPeriod;
                layericLatency *= clkPeriod;
                coreLatencyADC *= clkPeriod;
                coreLatencyAccum *= clkPeriod;
                coreLatencyOther *= clkPeriod;
            }
            
            double numTileOtherLayer = 0;
            double layerLeakageEnergy = 0;      
            for (int j=0; j<netStructure.size(); j++) {
                if (j != i) {
                    numTileOtherLayer += numTileEachLayer[0][j] * numTileEachLayer[1][j];
                }
            }
            // Anni update: other layer tiles and partial this layer tiles are in leakage
            layerLeakageEnergy = (numTileOtherLayer * tileLeakage + numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakageSRAMInUse) * layerReadLatency;
            
            cout << "layer" << i+1 << "'s readLatency is: " << layerReadLatency*1e9 << "ns" << endl;
            cout << "layer" << i+1 << "'s readDynamicEnergy is: " << layerReadDynamicEnergy*1e12 << "pJ" << endl;
            cout << "layer" << i+1 << "'s leakagePower is: " << numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage*1e6 << "uW" << endl;
            cout << "layer" << i+1 << "'s leakageEnergy is: " << layerLeakageEnergy*1e12 << "pJ" << endl;
            cout << "layer" << i+1 << "'s buffer latency is: " << layerbufferLatency*1e9 << "ns" << endl;
            cout << "layer" << i+1 << "'s buffer readDynamicEnergy is: " << layerbufferDynamicEnergy*1e12 << "pJ" << endl;
            cout << "layer" << i+1 << "'s ic latency is: " << layericLatency*1e9 << "ns" << endl;
            cout << "layer" << i+1 << "'s ic readDynamicEnergy is: " << layericDynamicEnergy*1e12 << "pJ" << endl;
            
            
            cout << endl;
            cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
            cout << endl;
            cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADC*1e9 << "ns" << endl;
            cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccum*1e9 << "ns" << endl;
            cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << coreLatencyOther*1e9 << "ns" << endl;
            cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADC*1e12 << "pJ" << endl;
            cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccum*1e12 << "pJ" << endl;
            cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << coreEnergyOther*1e12 << "pJ" << endl;
            cout << endl;
            cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
            cout << endl;
            
            chipReadLatency += layerReadLatency;
            chipReadDynamicEnergy += layerReadDynamicEnergy;
            chipLeakageEnergy += layerLeakageEnergy;
            chipLeakage += tileLeakage*numTileEachLayer[0][i] * numTileEachLayer[1][i];
            chipbufferLatency += layerbufferLatency;
            chipbufferReadDynamicEnergy += layerbufferDynamicEnergy;
            chipicLatency += layericLatency;
            chipicReadDynamicEnergy += layericDynamicEnergy;
            
            chipLatencyADC += coreLatencyADC;
            chipLatencyAccum += coreLatencyAccum;
            chipLatencyOther += coreLatencyOther;
            chipEnergyADC += coreEnergyADC;
            chipEnergyAccum += coreEnergyAccum;
            chipEnergyOther += coreEnergyOther;
        }
    } else {
        // pipeline system
        // 1) first pass: compute per-layer latency/energy and find final systemClock (bottleneck)
        double systemClock = 0.0;

        vector<double> readLatencyPerLayer;
        vector<double> readDynamicEnergyPerLayer;

        vector<double> bufferLatencyPerLayer;
        vector<double> bufferEnergyPerLayer;
        vector<double> icLatencyPerLayer;
        vector<double> icEnergyPerLayer;

        vector<double> coreLatencyADCPerLayer;
        vector<double> coreEnergyADCPerLayer;
        vector<double> coreLatencyAccumPerLayer;
        vector<double> coreEnergyAccumPerLayer;
        vector<double> coreLatencyOtherPerLayer;
        vector<double> coreEnergyOtherPerLayer;

        // [ADD] store leakage params per layer so we can compute leakage with final systemClock in 2nd pass
        vector<double> tileLeakagePerLayer;         // tileLeakage (W) per tile (or whatever unit your function returns)
        vector<double> tileLeakageInUsePerLayer;    // tileLeakageSRAMInUse per tile

        // reserve for speed (optional)
        const size_t L = netStructure.size();
        readLatencyPerLayer.reserve(L);
        readDynamicEnergyPerLayer.reserve(L);
        bufferLatencyPerLayer.reserve(L);
        bufferEnergyPerLayer.reserve(L);
        icLatencyPerLayer.reserve(L);
        icEnergyPerLayer.reserve(L);
        coreLatencyADCPerLayer.reserve(L);
        coreEnergyADCPerLayer.reserve(L);
        coreLatencyAccumPerLayer.reserve(L);
        coreEnergyAccumPerLayer.reserve(L);
        coreLatencyOtherPerLayer.reserve(L);
        coreEnergyOtherPerLayer.reserve(L);
        tileLeakagePerLayer.reserve(L);
        tileLeakageInUsePerLayer.reserve(L);

        // bottleneck info (already declared above in your code; here we just ensure reset)
        bottleneckLayer = -1;
        bottleneckReadLatency = 0.0;
        bottleneckBufferLatency = 0.0;
        bottleneckICLatency = 0.0;
        bottleneckCoreLatencyADC = 0.0;
        bottleneckCoreLatencyAccum = 0.0;
        bottleneckCoreLatencyOther = 0.0;

        for (int i = 0; i < (int)netStructure.size(); i++) {
            ChipCalculatePerformance(inputParameter, tech, cell, i,
                        argv[2*i+6], argv[2*i+6], argv[2*i+7], netStructure[i][6],
                        netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer_part, tileLocaEachLayer,
                        numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM,
                        CMTileheight, CMTilewidth, NMTileheight, NMTilewidth,
                        &layerReadLatency, &layerReadDynamicEnergy,
                        &tileLeakage, &tileLeakageSRAMInUse,
                        &layerbufferLatency, &layerbufferDynamicEnergy,
                        &layericLatency, &layericDynamicEnergy,
                        &coreLatencyADC, &coreLatencyAccum, &coreLatencyOther,
                        &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther,
                        false, &layerclkPeriod);

            if (param->synchronous) {
                layerReadLatency   *= clkPeriod;
                layerbufferLatency *= clkPeriod;
                layericLatency     *= clkPeriod;
                coreLatencyADC     *= clkPeriod;
                coreLatencyAccum   *= clkPeriod;
                coreLatencyOther   *= clkPeriod;
            }

            // store per-layer metrics
            readLatencyPerLayer.push_back(layerReadLatency);
            readDynamicEnergyPerLayer.push_back(layerReadDynamicEnergy);

            bufferLatencyPerLayer.push_back(layerbufferLatency);
            bufferEnergyPerLayer.push_back(layerbufferDynamicEnergy);
            icLatencyPerLayer.push_back(layericLatency);
            icEnergyPerLayer.push_back(layericDynamicEnergy);

            coreLatencyADCPerLayer.push_back(coreLatencyADC);
            coreEnergyADCPerLayer.push_back(coreEnergyADC);
            coreLatencyAccumPerLayer.push_back(coreLatencyAccum);
            coreEnergyAccumPerLayer.push_back(coreEnergyAccum);
            coreLatencyOtherPerLayer.push_back(coreLatencyOther);
            coreEnergyOtherPerLayer.push_back(coreEnergyOther);

            tileLeakagePerLayer.push_back(tileLeakage);
            tileLeakageInUsePerLayer.push_back(tileLeakageSRAMInUse);

            // update systemClock / bottleneck
            if (layerReadLatency > systemClock) {
                systemClock = layerReadLatency;

                bottleneckLayer = i;
                bottleneckReadLatency = layerReadLatency;
                bottleneckBufferLatency = layerbufferLatency;
                bottleneckICLatency = layericLatency;
                bottleneckCoreLatencyADC = coreLatencyADC;
                bottleneckCoreLatencyAccum = coreLatencyAccum;
                bottleneckCoreLatencyOther = coreLatencyOther;
            }
        }

        // 2) second pass: compute leakagePowerPerLayer using FINAL systemClock
        vector<double> leakagePowerPerLayer;
        leakagePowerPerLayer.reserve(L);

        for (int i = 0; i < (int)netStructure.size(); i++) {
            // average leakage power considering read latency (original intent)
            // P_avg = Ntile * ( P_idle*(T - t_read) + P_inuse*t_read ) / T
            // where T = systemClock, t_read = layerReadLatency
            const double nTiles = numTileEachLayer[0][i] * numTileEachLayer[1][i];
            const double tRead  = readLatencyPerLayer[i];

            double Pavg = 0.0;
            if (systemClock > 0) {
                Pavg = nTiles * ( tileLeakagePerLayer[i] * (systemClock - tRead)
                                + tileLeakageInUsePerLayer[i] * tRead ) / systemClock;
            }
            leakagePowerPerLayer.push_back(Pavg);
        }

        // [PRINT] bottleneck layer breakdown (optional, same as your previous block)
        cout << "================ Pipeline Bottleneck Layer =================" << endl;
        if (bottleneckLayer >= 0) {
            cout << "Bottleneck layer (defines system clock): layer" << bottleneckLayer+1
                 << " (readLatency=" << bottleneckReadLatency*1e9 << "ns)" << endl;
            cout << "  - buffer latency: " << bottleneckBufferLatency*1e9 << "ns" << endl;
            cout << "  - ic latency:     " << bottleneckICLatency*1e9 << "ns" << endl;
            cout << "  - ADC latency:    " << bottleneckCoreLatencyADC*1e9 << "ns" << endl;
            cout << "  - Accum latency:  " << bottleneckCoreLatencyAccum*1e9 << "ns" << endl;
            cout << "  - Other latency:  " << bottleneckCoreLatencyOther*1e9 << "ns" << endl;
        } else {
            cout << "WARNING: bottleneckLayer was not set (netStructure might be empty)." << endl;
        }
        cout << "===========================================================" << endl;

        // 3) reporting + chip-level aggregation (same as your existing loop, but now leakage uses leakagePowerPerLayer[i])
        for (int i = 0; i < (int)netStructure.size(); i++) {

            cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;

            cout << "layer" << i+1 << "'s readLatency is: " << readLatencyPerLayer[i]*1e9 << "ns" << endl;
            cout << "layer" << i+1 << "'s readDynamicEnergy is: " << readDynamicEnergyPerLayer[i]*1e12 << "pJ" << endl;
            cout << "layer" << i+1 << "'s leakagePower is: " << leakagePowerPerLayer[i]*1e6 << "uW" << endl;
            cout << "layer" << i+1 << "'s leakageEnergy is: " << leakagePowerPerLayer[i] * systemClock * 1e12 << "pJ" << endl;

            cout << "layer" << i+1 << "'s buffer latency is: " << bufferLatencyPerLayer[i]*1e9 << "ns" << endl;
            cout << "layer" << i+1 << "'s buffer readDynamicEnergy is: " << bufferEnergyPerLayer[i]*1e12 << "pJ" << endl;
            cout << "layer" << i+1 << "'s ic latency is: " << icLatencyPerLayer[i]*1e9 << "ns" << endl;
            cout << "layer" << i+1 << "'s ic readDynamicEnergy is: " << icEnergyPerLayer[i]*1e12 << "pJ" << endl;

            cout << endl;
            cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
            cout << endl;
            cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADCPerLayer[i]*1e9 << "ns" << endl;
            cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccumPerLayer[i]*1e9 << "ns" << endl;
            cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << coreLatencyOtherPerLayer[i]*1e9 << "ns" << endl;
            cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADCPerLayer[i]*1e12 << "pJ" << endl;
            cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccumPerLayer[i]*1e12 << "pJ" << endl;
            cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << coreEnergyOtherPerLayer[i]*1e12 << "pJ" << endl;
            cout << endl;
            cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
            cout << endl;

            // chip-level aggregation (pipeline semantics)
            chipReadLatency = systemClock;
            chipReadDynamicEnergy += readDynamicEnergyPerLayer[i];

            chipLeakageEnergy += leakagePowerPerLayer[i] * systemClock;
            chipLeakage += leakagePowerPerLayer[i];

            chipbufferLatency = MAX(chipbufferLatency, bufferLatencyPerLayer[i]);
            chipbufferReadDynamicEnergy += bufferEnergyPerLayer[i];

            chipicLatency = MAX(chipicLatency, icLatencyPerLayer[i]);
            chipicReadDynamicEnergy += icEnergyPerLayer[i];

            chipLatencyADC = MAX(chipLatencyADC, coreLatencyADCPerLayer[i]);
            chipLatencyAccum = MAX(chipLatencyAccum, coreLatencyAccumPerLayer[i]);
            chipLatencyOther = MAX(chipLatencyOther, coreLatencyOtherPerLayer[i]);

            chipEnergyADC += coreEnergyADCPerLayer[i];
            chipEnergyAccum += coreEnergyAccumPerLayer[i];
            chipEnergyOther += coreEnergyOtherPerLayer[i];
        }
    }

    
    cout << "------------------------------ Summary --------------------------------" <<  endl;
    cout << endl;
    cout << "ChipArea (e+7um^2) :" << endl;
    cout << chipArea * 1e5 << endl;
    cout << "Chip total CIM array (e+7um^2) :" << endl;
    cout << chipAreaArray * 1e5 << endl;
    cout << "Total IC Area on chip (Global and Tile/PE local) (e+7um^2) :" << endl;
    cout << chipAreaIC * 1e5 << endl;
    cout << "Total ADC (or S/As and precharger for SRAM) Area on chip (e+7um^2) :" << endl;
    cout << chipAreaADC * 1e5 << endl;
    cout << "Total Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) on chip (e+7um^2) :" << endl;
    cout << chipAreaAccum * 1e5 << endl;
    cout << "Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, pooling and activation units) (e+7um^2) :" << endl;
    cout << chipAreaOther * 1e5 << endl;
    cout << endl;


    if (! param->pipeline) {
        if (param->synchronous) cout << "Chip clock period is: " << clkPeriod*1e9 << "ns" <<endl;
        cout << "Chip layer-by-layer readLatency (per image) is: " << chipReadLatency*1e9 << "ns" << endl;
        cout << "Chip total readDynamicEnergy is: " << chipReadDynamicEnergy*1e12 << "pJ" << endl;
        cout << "Chip total leakage Energy is: " << chipLeakageEnergy*1e12 << "pJ" << endl;
        cout << "Chip total leakage Power is: " << chipLeakage*1e6 << "uW" << endl;
        cout << "Chip buffer readLatency is: " << chipbufferLatency*1e9 << "ns" << endl;
        cout << "Chip buffer readDynamicEnergy is: " << chipbufferReadDynamicEnergy*1e12 << "pJ" << endl;
        cout << "Chip ic readLatency is: " << chipicLatency*1e9 << "ns" << endl;
        cout << "Chip ic readDynamicEnergy is: " << chipicReadDynamicEnergy*1e12 << "pJ" << endl;
    } else {
        if (param->synchronous) cout << "Chip clock period is: " << clkPeriod*1e9 << "ns" <<endl;
        cout << "Chip pipeline-system-clock-cycle (per image) is: " << chipReadLatency*1e9 << "ns" << endl;
        cout << "Chip pipeline-system readDynamicEnergy (per image) is: " << chipReadDynamicEnergy*1e12 << "pJ" << endl;
        cout << "Chip pipeline-system leakage Energy (per image) is: " << chipLeakageEnergy*1e12 << "pJ" << endl;
        cout << "Chip pipeline-system leakage Power (per image) is: " << chipLeakage*1e6 << "uW" << endl;
        cout << "Chip pipeline-system buffer readLatency (per image) is: " << chipbufferLatency*1e9 << "ns" << endl;
        cout << "Chip pipeline-system buffer readDynamicEnergy (per image) is: " << chipbufferReadDynamicEnergy*1e12 << "pJ" << endl;
        cout << "Chip pipeline-system ic readLatency (per image) is: " << chipicLatency*1e9 << "ns" << endl;
        cout << "Chip pipeline-system ic readDynamicEnergy (per image) is: " << chipicReadDynamicEnergy*1e12 << "pJ" << endl;
    }
    
    cout << endl;
    cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
    cout << endl;
    cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << chipLatencyADC*1e9 << "ns" << endl;
    cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << chipLatencyAccum*1e9 << "ns" << endl;
    cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << chipLatencyOther*1e9 << "ns" << endl;
    cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << chipEnergyADC*1e12 << "pJ" << endl;
    cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << chipEnergyAccum*1e12 << "pJ" << endl;
    cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << chipEnergyOther*1e12 << "pJ" << endl;
    cout << endl;
    cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
    cout << endl;
    
    cout << endl;
    cout << "----------------------------- Performance -------------------------------" << endl;
    if (! param->pipeline) {
        if(param->validated){
            cout << "Energy Efficiency TOPS/W (Layer-by-Layer Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)/param->zeta << endl;    // post-layout energy increase, zeta = 1.23 by default
        }else{
            cout << "Energy Efficiency TOPS/W (Layer-by-Layer Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12) << endl;
        }
        cout << "Throughput TOPS (Layer-by-Layer Process): " << numComputation/(chipReadLatency*1e12) << endl;
        cout << "Throughput FPS (Layer-by-Layer Process): " << 1/(chipReadLatency) << endl;
        cout << "Compute efficiency TOPS/mm^2 (Layer-by-Layer Process): " << numComputation/(chipReadLatency*1e12)/(chipArea*1e6) << endl;
    } else {
        if(param->validated){
            cout << "Energy Efficiency TOPS/W (Pipelined Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)/param->zeta << endl; // post-layout energy increase, zeta = 1.23 by default
        }else{
            cout << "Energy Efficiency TOPS/W (Pipelined Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12) << endl;
        }
        cout << "Throughput TOPS (Pipelined Process): " << numComputation/(chipReadLatency*1e12) << endl;
        cout << "Throughput FPS (Pipelined Process): " << 1/(chipReadLatency) << endl;
        cout << "Compute efficiency TOPS/mm^2 (Pipelined Process): " << numComputation/(chipReadLatency*1e12)/(chipArea*1e6) << endl;
    }
    cout << "-------------------------------------- Hardware Performance Done --------------------------------------" <<  endl;
    cout << endl;
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(stop-start);
    cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
    cout << "Total Run-time of NeuroSim: " << duration.count() << " seconds" << endl;
    cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
    
    // debugging code

    /*
    fstream read;
    // read.open("filelocation/filename",fstream::app);    
    read.open("/home/junmo/DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM/Data_TechnologyUpdate/overall_metrics.csv",fstream::app); 
    
    // enter the filelocation/filename where you want to store the printed values. 

    read<<param->technode<<", " ;
    read<<param->operationmode<<", ";
    read<<param->memcelltype<<", ";
    read<<param->accesstype<<", ";

    read<<chipArea*1e12<<", "<<chipAreaArray*1e12<<", "<<chipAreaIC*1e12<<", ";
    read<<chipAreaADC*1e12<<", "<<chipAreaAccum*1e12<<", "<<chipAreaOther*1e12<<", ";
    read<<clkPeriod*1e9<<", ";
    read<<chipReadLatency*1e9<<", ";
    read<<chipReadDynamicEnergy*1e12<<", ";
    read<<chipLeakageEnergy*1e12<<", ";
    read<<chipLeakage*1e6<<", ";

    read<<chipbufferLatency*1e9<<", ";
    read<<chipbufferReadDynamicEnergy*1e12<<", ";
    read<<chipicLatency*1e9 <<", ";
    read<<chipicReadDynamicEnergy*1e12 <<", ";

    read<<chipLatencyADC*1e9<<", ";
    read<<chipLatencyAccum*1e9 <<", ";
    read<<chipLatencyOther*1e9 <<", ";
    read<<chipEnergyADC*1e12 <<", ";
    read<<chipEnergyAccum*1e12<<", ";
    read<<chipEnergyOther*1e12<<", ";

    read<<numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)/param->zeta<<", ";
    read<<numComputation/(chipReadLatency*1e12)<<", ";
    read<<1/(chipReadLatency) <<", ";
    read<<numComputation/(chipReadLatency*1e12)/(chipArea*1e6) <<", ";
    read<<numComputation<<", ";
    read<< "param->inputtoggle"<<", "<<param->inputtoggle<<", ";
    read<< "param->numRowParallel"<<", "<<param->numRowParallel<<", ";
    read<< "onoff"<<", "<<param->resistanceOn/param->resistanceOff<<", ";
    read<< "levelOutput"<<", "<<param->levelOutput <<", ";
    read<< "CellBit"<<", "<<param->cellBit <<", ";
    read<< "ADCcurrentmode" <<", "<<param->currentMode<<", ";
    read<< "SubArraySize" <<", "<<param->numRowSubArray<<", ";
    read<< "ADCdelay" <<", "<<param->ADClatency<<", ";
    read<< "rowdelay" <<", "<<param->rowdelay<<", ";
    read<< "muxdelay" <<", "<<param->muxdelay<<", ";
    read<<endl;

    */


    /************* Save Summary CSV (rows keyed by SynapseBit) *************/
    {
        // 현재 날짜(YYYYMMDD) 가져오기
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        ss << std::put_time(std::localtime(&t), "%Y%m%d");

        std::string cellTypeStr = std::to_string(param->memcelltype);
        std::string synBitStr  = std::to_string(static_cast<int>(param->synapseBit));

        // 저장 경로
        const std::string base_dir = ".";
        const std::string out_dir  = base_dir + "/summary";
        const std::string out_csv  = out_dir + "/summary_Cell" + cellTypeStr + 
                                     "_SynBit" + synBitStr + "_" + ss.str() + ".csv";

        // 디렉토리 생성
        std::string mkcmd = "mkdir -p \"" + out_dir + "\"";
        (void)system(mkcmd.c_str());

        // ---------- 스케일 변환 헬퍼 (SI -> 원하는 단위) ----------
        // (가정) NeuroSim 내부 출력:
        //  - area : m^2
        //  - time : s
        //  - energy: J
        auto AREA_MM2 = [](double area_m2){ return area_m2 * 1e6; }; // m^2 -> mm^2
        auto TIME_US  = [](double sec){     return sec     * 1e6; }; // s   -> us
        auto ENE_UJ   = [](double J){       return J       * 1e6; }; // J   -> uJ

        // ---------- 공통 요약 (면적) ----------
        const double m_ChipArea_mm2       = AREA_MM2(chipArea);
        const double m_ChipAreaArray_mm2  = AREA_MM2(chipAreaArray);
        const double m_ChipAreaIC_mm2     = AREA_MM2(chipAreaIC);
        const double m_ChipAreaADC_mm2    = AREA_MM2(chipAreaADC);
        const double m_ChipAreaAccum_mm2  = AREA_MM2(chipAreaAccum);
        const double m_ChipAreaOther_mm2  = AREA_MM2(chipAreaOther);

        // ---------- 공통: 클럭/지연/에너지/파워 ----------
        const double m_ClkPeriod_us       = (param->synchronous ? TIME_US(clkPeriod) : 0.0);

        // 레이어바이레이어 vs 파이프라인 공통에서 쓰는 기본 단위 변환
        const double m_ReadLatency_us     = TIME_US(chipReadLatency);
        const double m_ReadDynEnergy_uJ   = ENE_UJ(chipReadDynamicEnergy);
        const double m_LeakageEnergy_uJ   = ENE_UJ(chipLeakageEnergy);
        const double m_LeakagePower_uW    = chipLeakage * 1e6; // uW (파워는 기존 요청대로 유지)

        const double m_BufferLatency_us   = TIME_US(chipbufferLatency);
        const double m_BufferDynEnergy_uJ = ENE_UJ(chipbufferReadDynamicEnergy);
        const double m_IcLatency_us       = TIME_US(chipicLatency);
        const double m_IcDynEnergy_uJ     = ENE_UJ(chipicReadDynamicEnergy);

        // 브레이크다운 (ADC/Accum/Other)
        const double m_Lat_ADC_us         = TIME_US(chipLatencyADC);
        const double m_Lat_Accum_us       = TIME_US(chipLatencyAccum);
        const double m_Lat_Other_us       = TIME_US(chipLatencyOther);
        const double m_Ene_ADC_uJ         = ENE_UJ(chipEnergyADC);
        const double m_Ene_Accum_uJ       = ENE_UJ(chipEnergyAccum);
        const double m_Ene_Other_uJ       = ENE_UJ(chipEnergyOther);

        // ---------- [ADD] Pipeline bottleneck breakdown (ns) ----------
        // NOTE: These variables must be computed in the pipeline section before this CSV block.
        // If not in pipeline mode, we will leave them blank by not writing to table.
        // ---------- Pipeline bottleneck breakdown (us) ----------
        const double m_BN_ReadLatency_us   = TIME_US(bottleneckReadLatency);
        const double m_BN_BufferLatency_us = TIME_US(bottleneckBufferLatency);
        const double m_BN_IcLatency_us     = TIME_US(bottleneckICLatency);
        const double m_BN_ADCLatency_us    = TIME_US(bottleneckCoreLatencyADC);
        const double m_BN_AccumLatency_us  = TIME_US(bottleneckCoreLatencyAccum);
        const double m_BN_OtherLatency_us  = TIME_US(bottleneckCoreLatencyOther);

        const double m_BN_SumLatency_us =
            m_BN_BufferLatency_us + m_BN_IcLatency_us +
            m_BN_ADCLatency_us + m_BN_AccumLatency_us + m_BN_OtherLatency_us;

        // 성능 (TOPS/W, TOPS, FPS, TOPS/mm^2)
        double m_EE_TOPSW = 0.0;
        {
            // EnergyEfficiency는 J 기반 계산을 그대로 두고, denominator를 "J"로 구성
            // (validated이면 zeta 반영)
            const double e_J = (chipReadDynamicEnergy + chipLeakageEnergy);
            const double denom = param->validated ? (e_J * param->zeta) : e_J;
            if (denom > 0) m_EE_TOPSW = numComputation / (denom * 1e12); // TOPS/W = ops/s / W
            // 주의: numComputation이 "op count" 단위이고, 기존 코드가 pJ 기반으로 TOPS/W를 만들었다면
            //      원래 정의를 유지하려면 아래 '원래 방식' 블록으로 교체하세요.
        }

        // --- 원래 코드의 TOPS/W 정의(형태) 유지 버전이 필요하면 이걸 쓰세요 ---
        // double m_EE_TOPSW = 0.0;
        // {
        //     const double e_pJ = (chipReadDynamicEnergy*1e12 + chipLeakageEnergy*1e12); // pJ
        //     const double denom = param->validated ? (e_pJ * param->zeta) : e_pJ;
        //     if (denom > 0) m_EE_TOPSW = numComputation / denom;
        // }

        const double m_Throughput_TOPS  = numComputation / (chipReadLatency * 1e12);
        const double m_FPS              = 1.0 / chipReadLatency;
        const double m_CompEff_TOPSmm2  = (chipArea > 0) ? (m_Throughput_TOPS / (chipArea * 1e6)) : 0.0; // chipArea[m^2]->mm^2

        // 시뮬레이션 런타임(초 -> us)
        const double m_Runtime_us = TIME_US(static_cast<double>(duration.count()));

        // ---------- 표 구조: rows = metric name, cols = synapseBit ----------
        const int synBit = static_cast<int>(param->synapseBit);

        // 행 이름(모드별로 구분된 라벨 포함)
        const std::vector<std::string> METRICS = {
            // Summary (Area)
            "ChipArea_(mm^2)",
            "ChipArea_CIMarray_(mm^2)",
            "ChipArea_IC_(mm^2)",
            "ChipArea_ADC_(mm^2)",
            "ChipArea_Accum_(mm^2)",
            "ChipArea_Other_(mm^2)",

            // Common timing/energy/power
            "ClockPeriod_(us)",

            // Layer-by-Layer
            "LBL_ReadLatency_(us)",
            "LBL_ReadDynamicEnergy_(uJ)",
            "LBL_LeakageEnergy_(uJ)",
            "LBL_LeakagePower_(uW)",
            "LBL_BufferLatency_(us)",
            "LBL_BufferReadDynamicEnergy_(uJ)",
            "LBL_IcLatency_(us)",
            "LBL_IcReadDynamicEnergy_(uJ)",

            // Pipelined
            "PIPE_ReadLatency_(us)",
            "PIPE_ReadDynamicEnergy_(uJ)",
            "PIPE_LeakageEnergy_(uJ)",
            "PIPE_LeakagePower_(uW)",
            "PIPE_BufferLatency_(us)",
            "PIPE_BufferReadDynamicEnergy_(uJ)",
            "PIPE_IcLatency_(us)",
            "PIPE_IcReadDynamicEnergy_(uJ)",

            // Breakdown (공용)
            "Breakdown_ADC_Latency_(us)",
            "Breakdown_Accum_Latency_(us)",
            "Breakdown_Other_Latency_(us)",
            "Breakdown_ADC_Energy_(uJ)",
            "Breakdown_Accum_Energy_(uJ)",
            "Breakdown_Other_Energy_(uJ)",

            // Performance (모드별 레이블)
            "LBL_EnergyEfficiency_(TOPS/W)",
            "LBL_Throughput_(TOPS)",
            "LBL_Throughput_(FPS)",
            "LBL_ComputeEfficiency_(TOPS/mm^2)",
            "PIPE_EnergyEfficiency_(TOPS/W)",
            "PIPE_Throughput_(TOPS)",
            "PIPE_Throughput_(FPS)",
            "PIPE_ComputeEfficiency_(TOPS/mm^2)",

            "PIPE_BottleneckLayer_(index1based)",
            "PIPE_BottleneckReadLatency_(us)",
            "PIPE_BottleneckBufferLatency_(us)",
            "PIPE_BottleneckIcLatency_(us)",
            "PIPE_BottleneckADCLatency_(us)",
            "PIPE_BottleneckAccumLatency_(us)",
            "PIPE_BottleneckOtherLatency_(us)",
            "PIPE_BottleneckBreakdownSum_(us)",
            
            // Simulation
            "Simulation_Runtime_(us)"
        };

        // metric -> (synapseBit -> value)
        std::map<std::string, std::map<int,double>> table;
        std::set<int> synBits; // CSV에 포함할 synapseBit 집합

        // ---- 기존 CSV 로드 (있다면) ----
        {
            std::ifstream fin(out_csv);
            if (fin.good()) {
                std::string header;

                // 1) Metadata 블록 스킵: 첫 "Metric,..." 헤더까지 건너뜀
                while (std::getline(fin, header)) {
                    if (header.rfind("Metric", 0) == 0) break; // line starts with "Metric"
                }

                if (fin.good() && header.rfind("Metric", 0) == 0) {
                    // 헤더 파싱: "Metric,SynapseBit4,SynapseBit6,..."
                    std::vector<std::string> cols;
                    {
                        std::stringstream hs(header);
                        std::string c;
                        while (std::getline(hs, c, ',')) cols.push_back(c);
                    }

                    std::map<size_t, int> colToSB;
                    for (size_t i = 1; i < cols.size(); ++i) {
                        const std::string &name = cols[i];
                        std::string digits;
                        for (char ch : name)
                            if (std::isdigit(static_cast<unsigned char>(ch)))
                                digits.push_back(ch);
                        if (!digits.empty()) {
                            int sb = std::stoi(digits);
                            colToSB[i] = sb;
                            synBits.insert(sb);
                        }
                    }

                    // 본문 읽기
                    std::string line;
                    while (std::getline(fin, line)) {
                        if (line.empty()) continue;

                        std::stringstream ls(line);
                        std::vector<std::string> toks;
                        std::string cell;
                        while (std::getline(ls, cell, ',')) toks.push_back(cell);
                        if (toks.empty()) continue;

                        const std::string metric = toks[0];
                        for (size_t i = 1; i < toks.size(); ++i) {
                            auto it = colToSB.find(i);
                            if (it == colToSB.end()) continue;
                            if (toks[i].empty()) continue;
                            try {
                                table[metric][it->second] = std::stod(toks[i]);
                            } catch (...) {}
                        }
                    }
                }
                fin.close();
            }
        }

        // ---- 현재 실행 synapseBit 포함 ----
        synBits.insert(synBit);

        // ---- 값 채우기: 요약(면적) ----
        table["ChipArea_(mm^2)"][synBit]            = m_ChipArea_mm2;
        table["ChipArea_CIMarray_(mm^2)"][synBit]   = m_ChipAreaArray_mm2;
        table["ChipArea_IC_(mm^2)"][synBit]         = m_ChipAreaIC_mm2;
        table["ChipArea_ADC_(mm^2)"][synBit]        = m_ChipAreaADC_mm2;
        table["ChipArea_Accum_(mm^2)"][synBit]      = m_ChipAreaAccum_mm2;
        table["ChipArea_Other_(mm^2)"][synBit]      = m_ChipAreaOther_mm2;

        // ---- 공통 (클럭) ----
        if (param->synchronous) {
            table["ClockPeriod_(us)"][synBit] = m_ClkPeriod_us;
        }

        // ---- 모드별: LBL vs PIPE (둘 중 하나만 채움) ----
        if (!param->pipeline) {
            // Layer-by-Layer
            table["LBL_ReadLatency_(us)"][synBit]             = m_ReadLatency_us;
            table["LBL_ReadDynamicEnergy_(uJ)"][synBit]       = m_ReadDynEnergy_uJ;
            table["LBL_LeakageEnergy_(uJ)"][synBit]           = m_LeakageEnergy_uJ;
            table["LBL_LeakagePower_(uW)"][synBit]            = m_LeakagePower_uW;
            table["LBL_BufferLatency_(us)"][synBit]           = m_BufferLatency_us;
            table["LBL_BufferReadDynamicEnergy_(uJ)"][synBit] = m_BufferDynEnergy_uJ;
            table["LBL_IcLatency_(us)"][synBit]               = m_IcLatency_us;
            table["LBL_IcReadDynamicEnergy_(uJ)"][synBit]     = m_IcDynEnergy_uJ;

            table["LBL_EnergyEfficiency_(TOPS/W)"][synBit]    = m_EE_TOPSW;
            table["LBL_Throughput_(TOPS)"][synBit]            = m_Throughput_TOPS;
            table["LBL_Throughput_(FPS)"][synBit]             = m_FPS;
            table["LBL_ComputeEfficiency_(TOPS/mm^2)"][synBit] = m_CompEff_TOPSmm2;
        } else {
            // Pipelined
            table["PIPE_ReadLatency_(us)"][synBit]             = m_ReadLatency_us;
            table["PIPE_ReadDynamicEnergy_(uJ)"][synBit]       = m_ReadDynEnergy_uJ;
            table["PIPE_LeakageEnergy_(uJ)"][synBit]           = m_LeakageEnergy_uJ;
            table["PIPE_LeakagePower_(uW)"][synBit]            = m_LeakagePower_uW;
            table["PIPE_BufferLatency_(us)"][synBit]           = m_BufferLatency_us;
            table["PIPE_BufferReadDynamicEnergy_(uJ)"][synBit] = m_BufferDynEnergy_uJ;
            table["PIPE_IcLatency_(us)"][synBit]               = m_IcLatency_us;
            table["PIPE_IcReadDynamicEnergy_(uJ)"][synBit]     = m_IcDynEnergy_uJ;

            table["PIPE_EnergyEfficiency_(TOPS/W)"][synBit]    = m_EE_TOPSW;
            table["PIPE_Throughput_(TOPS)"][synBit]            = m_Throughput_TOPS;
            table["PIPE_Throughput_(FPS)"][synBit]             = m_FPS;
            table["PIPE_ComputeEfficiency_(TOPS/mm^2)"][synBit] = m_CompEff_TOPSmm2;
        
            table["PIPE_BottleneckLayer_(index1based)"][synBit] = static_cast<double>(bottleneckLayer + 1);
            table["PIPE_BottleneckReadLatency_(us)"][synBit]    = m_BN_ReadLatency_us;
            table["PIPE_BottleneckBufferLatency_(us)"][synBit]  = m_BN_BufferLatency_us;
            table["PIPE_BottleneckIcLatency_(us)"][synBit]      = m_BN_IcLatency_us;
            table["PIPE_BottleneckADCLatency_(us)"][synBit]     = m_BN_ADCLatency_us;
            table["PIPE_BottleneckAccumLatency_(us)"][synBit]   = m_BN_AccumLatency_us;
            table["PIPE_BottleneckOtherLatency_(us)"][synBit]   = m_BN_OtherLatency_us;
            table["PIPE_BottleneckBreakdownSum_(us)"][synBit]   = m_BN_SumLatency_us;

        }

        // ---- Breakdown (공용) ----
        table["Breakdown_ADC_Latency_(us)"][synBit]   = m_Lat_ADC_us;
        table["Breakdown_Accum_Latency_(us)"][synBit] = m_Lat_Accum_us;
        table["Breakdown_Other_Latency_(us)"][synBit] = m_Lat_Other_us;
        table["Breakdown_ADC_Energy_(uJ)"][synBit]    = m_Ene_ADC_uJ;
        table["Breakdown_Accum_Energy_(uJ)"][synBit]  = m_Ene_Accum_uJ;
        table["Breakdown_Other_Energy_(uJ)"][synBit]  = m_Ene_Other_uJ;

        // ---- Simulation ----
        table["Simulation_Runtime_(us)"][synBit]      = m_Runtime_us;

        // ---- CSV 덮어쓰기(전체 재작성) ----
        std::ofstream csv(out_csv, std::ios::trunc);
        csv.setf(std::ios::fixed);
        csv.precision(6);

        // 하드웨어 파라미터 (Metadata) 출력
        csv << "[Hardware Configuration]\n";
        csv << "Technology Node (nm)," << param->technode << "\n";
        csv << "Memory Cell Type (1:SRAM 2:RRAM 3:FeFET)," << param->memcelltype << "\n";
        csv << "Read Voltage (V)," << param->readVoltage << "\n";
        csv << "Access Voltage (V)," << param->accessVoltage << "\n";
        csv << "Access Resistance (Ohm)," << param->resistanceAccess << "\n";
        csv << "Write Voltage (V)," << param->writeVoltage << "\n";
        csv << "numColMuxed," << param->numColMuxed << "\n";
        csv << "IR Drop Tolerance," << IR_DROP_TOLERANCE << "\n";
        csv << "\n";

        // SynapseBit 컬럼 정렬
        std::vector<int> synBitsSorted(synBits.begin(), synBits.end());
        std::sort(synBitsSorted.begin(), synBitsSorted.end());

        // 헤더 1회만 출력 (중복 제거)
        csv << "Metric";
        for (int sb : synBitsSorted) csv << ",SynapseBit" << sb;
        csv << "\n";

        // METRICS 순서대로 출력 (없는 값은 빈 칸)
        for (const auto &m : METRICS) {
            csv << m;
            for (int sb : synBitsSorted) {
                auto rIt = table.find(m);
                if (rIt != table.end()) {
                    auto cIt = rIt->second.find(sb);
                    if (cIt != rIt->second.end()) csv << "," << cIt->second;
                    else                          csv << ",";
                } else {
                    csv << ",";
                }
            }
            csv << "\n";
        }
        csv.close();
    }
    /*********** End Save Summary CSV *************/


    return 0;
}

vector<vector<double> > getNetStructure(const string &inputfile) {
    ifstream infile(inputfile.c_str());      
    string inputline;
    string inputval;
    
    int ROWin=0, COLin=0;      
    if (!infile.good()) {        
        cerr << "Error: the input file cannot be opened!" << endl;
        exit(1);
    }else{
        while (getline(infile, inputline, '\n')) {       
            ROWin++;                                
        }
        infile.clear();
        infile.seekg(0, ios::beg);      
        if (getline(infile, inputline, '\n')) {        
            istringstream iss (inputline);      
            while (getline(iss, inputval, ',')) {       
                COLin++;
            }
        }   
    }
    infile.clear();
    infile.seekg(0, ios::beg);          

    vector<vector<double> > netStructure;               
    for (int row=0; row<ROWin; row++) { 
        vector<double> netStructurerow;
        getline(infile, inputline, '\n');             
        istringstream iss;
        iss.str(inputline);
        for (int col=0; col<COLin; col++) {       
            while(getline(iss, inputval, ',')){ 
                istringstream fs;
                fs.str(inputval);
                double f=0;
                fs >> f;                
                netStructurerow.push_back(f);           
            }           
        }       
        netStructure.push_back(netStructurerow);
    }
    infile.close();
    
    return netStructure;
    netStructure.clear();
}   

