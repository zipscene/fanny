var libfanny = require('./build/Release/addon-floatfann');
var FANNY = libfanny.FANNY;

var fanny = new FANNY({
	layers: [ 2, 10, 5 ]
});

var inputs = [ 0.2, 0.8 ];
var results = fanny.run(inputs);

console.log(results);

fanny.runAsync(inputs, function(err, results) {
	console.log(err, results);
});

