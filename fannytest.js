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
console.log('GET MSE OF TRAINING DATA :: ' + fanny.testData(td));
console.log('SHOULD HAVE UPDATED MSE :: ' + fanny.getMSE());


td.saveTrain("boolean-logic-training-data.txt", function(err) {
	console.log('TD Save results', err);
});
var td3 = new TrainingData(td);
td3.scaleInputTrainData(1, 2);
console.log('SCALE INPUT TRAIN DATA', td3.getInput());
//FANNY.loadFile("/asdasd", function(e, r) {
//	console.log('loadFile result', e, r);
//});

fanny.save("/tmp/testfann", function(err) {
	console.log('FANN Save result', err);
});

setTimeout(function() { console.log('Done.'); }, 5000);
