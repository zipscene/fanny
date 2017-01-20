var libfanny = require('./build/Release/addon-floatfann');
var FANNY = libfanny.FANNY;

var fanny = new FANNY({
	layers: [ 2, 10, 1 ]
});

