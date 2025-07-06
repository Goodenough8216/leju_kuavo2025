// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Kuavo Manual',
  tagline: 'Kuavo\'s development manual',
  url: 'https://kuavo.lejurobot.com',
  baseUrl: process.env.BASE_URL || 'manual/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans'],
  },

  onBrokenLinks: 'warn', // or 'ignore', 'log', 'throw'

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: 'new-docs',
          routeBasePath: '/',
        },
      }),
    ],
  ],
};

module.exports = config;
