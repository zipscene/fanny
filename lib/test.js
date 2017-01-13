var fanny = require('../build/Release/addon');

var FANN = fanny.FANN;

var fann = new FANN();

var res = fann.test([ "foo", "bar", "baz" ]);
console.log(res);

