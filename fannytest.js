var libfanny = require('./build/Release/addon-floatfann');
var FANNY = libfanny.FANNY;
var TrainingData = libfanny.TrainingData;

var fanny = new FANNY({
	layers: [ 2, 20, 5 ]
});

console.log('NUM INPUT :: ' + fanny.getNumInput());
console.log('NUM OUTPUT :: ' + fanny.getNumOutput());
console.log('NUM NEURONS :: ' + fanny.getTotalNeurons());
console.log('NUM CONNECTIONS :: ' + fanny.getTotalConnections());
console.log('NUM LAYERS :: ' + fanny.getNumLayers());

var inputs = [ 0.2, 0.8 ];
var results = fanny.run(inputs);
console.log(results);

var fanny2 = new FANNY(fanny);
results = fanny2.run(inputs);
console.log(results);

var fanny3 = new FANNY(fanny);
var outputs = [ 0.1, 0.2, 0.3, 0.4, 0.5 ];
var preTrainResults = fanny3.run(inputs);
console.log('PRE TRAIN :: ', preTrainResults);
fanny3.train(inputs, outputs);
var postTrainResults = fanny3.run(inputs);
console.log('POST TRAIN :: ', postTrainResults);

var fanny4 = new FANNY(fanny)
var outputs = [ 0.1, 0.2, 0.3, 0.4, 0.5 ];
var preTestMSE = fanny4.getMSE();
console.log('PRE TEST MSE :: ' + preTestMSE);
var testOutput = fanny4.test(inputs, outputs);
console.log('TEST OUTPUT :: ', testOutput);
var postTestMSE = fanny4.getMSE();
console.log('POST TEST MSE :: ' + postTestMSE);


console.log('BIT FAIL :: ' + fanny.getBitFail());
console.log('GET MSE :: ' + fanny.getMSE());
console.log('GET LEARNING RATE :: ' + fanny.getLearningRate());
console.log('GET PROP DECAY :: ' + fanny.getQuickPropDecay());
console.log('GET PROP MU :: ' + fanny.getQuickPropMu());
console.log('GET RPROP INCREASE FACTOR :: ' + fanny.getRpropIncreaseFactor());
console.log('GET RPROP DECREASE FACTOR :: ' + fanny.getRpropDecreaseFactor());
console.log('GET RPROP DELTA ZERO :: ' + fanny.getRpropDeltaZero());
console.log('GET RPROP DELTA MIN :: ' + fanny.getRpropDeltaMin());
console.log('GET RPROP DELTA MAX :: '  + fanny.getRpropDeltaMax());
console.log('LAYERS ARRAY :: ' + fanny.getLayerArray().join(', '));
console.log('BIAS ARRAY :: ' + fanny.getBiasArray().join(', '));

//fanny.runAsync(inputs, function(err, results) {
//	console.log(err, results);
//});

var td = new TrainingData();
var td2 = new TrainingData(td);
td.setTrainData([
	[ 1, 0 ],
	[ 0, 1 ],
	[ 0, 0 ],
	[ 1, 1 ]
], [
	// AND, OR, NAND, NOR, XOR
	[ 0, 1, 1, 0, 1 ],
	[ 0, 1, 1, 0, 1 ],
	[ 0, 1, 1, 0, 0 ],
	[ 1, 1, 0, 0, 0 ]
]);

fanny.initWeights(td);
console.log('SHOULD HAVE UPDATED MSE :: ' + fanny.getMSE());


td.saveTrain("boolean-logic-training-data.txt", function(err) {
	console.log('TD Save results', err);
});
var td3 = new TrainingData(td);
td3.scaleInputTrainData(1, 2);
console.log('SCALE INPUT TRAIN DATA :: ' + td3.getInput());
td3.scaleOutputTrainData(1, 2);
console.log('SCALE OUTPUT TRAIN DATA :: ' + td3.getOutput());
td3.scaleTrainData(2, 3);
console.log('SCALE TRAIN DATA INPUT :: ', td3.getInput());
console.log('SCALE TRAIN DATA OUTPUT :: ', td3.getOutput());
td3.subsetTrainData(1, 1);
console.log('SUBSET TRAIN DATA INPUT :: ', td3.getInput());
console.log('SUBSET TRAIN DATA OUTPUT :: ', td3.getOutput());
//FANNY.loadFile("/asdasd", function(e, r) {
//	console.log('loadFile result', e, r);
//});

var fanny5 = new FANNY(fanny)
var td5 = new TrainingData();
td5.setTrainData([
	[ 2, 1 ],
	[ 1, 2 ],
	[ 1, 1 ],
	[ 2, 2 ]
], [
	// AND, OR, NAND, NOR, XOR
	[ 1, 2, 2, 1, 2 ],
	[ 1, 2, 2, 1, 2 ],
	[ 1, 2, 2, 1, 1 ],
	[ 2, 2, 1, 1, 1 ]
]);

// Test scaleTrain and set setScalingParams
fanny5.setScalingParams(td5, 2, 3, 2, 3);
fanny5.scaleTrain(td5);
console.log('SCALE TRAIN INPUT :: ', td5.getInput());
console.log('SCALE TRAIN OUTPUT :: ', td5.getOutput());
// Test descaleTrain
fanny5.descaleTrain(td5);
console.log('DESCALE TRAIN INPUT :: ', td5.getInput());
console.log('DESCALE TRAIN OUTPUT :: ', td5.getOutput());
// Test clearScaling Params
console.log('CLEAR SCALING PARAMS :: ' + fanny5.clearScalingParams());
// Test setInputScalingParams and setOutputScalingParams
fanny5.setInputScalingParams(td5, 3, 4);
fanny5.setOutputScalingParams(td5, 3, 4);
fanny5.scaleTrain(td5);
console.log('SCALE TRAIN INPUT :: ', td5.getInput());
console.log('SCALE TRAIN OUTPUT :: ', td5.getOutput());

/* fanny.save("/tmp/testfann", function(err) {
	console.log('FANN Save result', err);
	});*/

console.log('Training');
fanny.trainOnData(td, 1000, 1, 0.001, function(err, res) {
	console.log('Training result', err, res);
});

setTimeout(function() { console.log('Done.'); }, 5000);
