const path = require('path');
const { spawn } = require('child_process');

describe('Linter', function() {
	it('should not have any linter problems', function(done) {
		this.timeout(60000);

		let isFinished = false;

		const finish = (error) => {
			if (isFinished) return;

			isFinished = true;

			if (typeof error === 'number') {
				throw new Error(`Project contains linter errors (exit code ${error})`);
			} else if (error instanceof Error) {
				throw error;
			} else if (error) {
				throw new Error(`${error}`);
			}

			done();
		};

		let cwd = path.resolve(__dirname, '../..');
		let lintProc = spawn('node', [ path.resolve(cwd, 'node_modules/.bin/eslint'), '.' ], { cwd });

		lintProc.on('error', finish);

		lintProc.on('exit', (code, signal) => {
			if (code !== null) {
				finish(code || null);
			} else if (signal !== null) {
				finish(`Unexpected linter exit: ${signal}`);
			} else {
				finish('Unexpected linter exit');
			}
		});
	});
});
