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


console.log('BIT FAIL :: ' + fanny.getBitFail());
console.log('GET MSE :: ' + fanny.getMSE());
console.log('GET LEARNING RATE :: ' + fanny.getLearningRate());
console.log('GET PROP DECAY :: ' + fanny.getQuickPropDecay());
console.log('GET PROP MU :: ' + fanny.getQuickPropMu());
console.log('GET RPROP INCREASE FACTOR :: ' + fanny.getRpropIncreaseFactor());
console.log('GET RPROP DECREASE FACTOR :: ' + fanny.getRpropDecreaseFactor());
console.log('GET RPROP DELTA ZERO :: ' + fanny.getRpropDeltaZero());
console.log('GET RPROP DELTA MIN :: ' + fanny.getRpropDeltaMin());
console.log('GET RPROP DELTA MAX :: ' + fanny.getRpropDeltaMax());

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

fanny.save("/tmp/testfann", function(err) {
	console.log('FANN Save result', err);
});

setTimeout(function() { console.log('Done.'); }, 5000);
