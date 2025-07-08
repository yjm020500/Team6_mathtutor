const mathjax = require('mathjax-node');
const sre = require('speech-rule-engine');

mathjax.config({
  MathJax: {
    // 기본 설정 그대로 사용
  }
});
mathjax.start();

async function main() {
  sre.setupEngine({
  locale: 'ko',
  domain: 'math',
  speech: 'mathspeak',
  modality: 'speech',
  style: 'default',  // 'default' or 'brief'
  structure: true,
  speechAttributes: ['clearspeak'],
  // 등 필요에 따라 추가 설정 가능
});

  await sre.engineReady();

  let input = '';

  process.stdin.setEncoding('utf8');
  process.stdin.on('data', chunk => {
    input += chunk;
  });

  process.stdin.on('end', () => {
    const latex = input.trim();
    console.log("[입력 LaTeX]:", latex);

    mathjax.typeset({
      math: latex,
      format: "TeX",
      mml: true
    }, (data) => {
      if (data.errors) {
        console.error("MathJax 변환 에러:", data.errors);
        process.exit(1);
      }
      try {
        const mathml = data.mml;
        console.log("[변환된 MathML]:", mathml);

        const speech = sre.toSpeech(mathml);
        console.log("[출력 음성 텍스트]:", speech);

        console.log(speech);
      } catch (err) {
        console.error('SRE 변환 에러:', err.message);
        process.exit(1);
      }
    });
  });
}

main();