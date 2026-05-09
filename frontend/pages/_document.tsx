import Document, { Html, Head, Main, NextScript } from 'next/document';

export default class MyDocument extends Document {
  render() {
    return (
      <Html lang="sw">
        <Head>
          <meta name="description" content="Real-time Tanzania Sign Language detection and translation powered by AI" />
          <meta name="keywords" content="sign language,TSL,Tanzania,AI,translation,accessibility" />
        </Head>
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    );
  }
}
