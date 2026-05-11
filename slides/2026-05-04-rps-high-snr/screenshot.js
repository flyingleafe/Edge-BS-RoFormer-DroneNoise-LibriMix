const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  
  for (let i = 1; i <= 5; i++) {
    await page.goto(`http://localhost:3030/${i}`);
    await page.waitForTimeout(3000);
    await page.screenshot({ path: `slide_${String(i).padStart(2, '0')}.png`, fullPage: false });
    console.log(`Captured slide ${i}`);
  }
  
  await browser.close();
})();
