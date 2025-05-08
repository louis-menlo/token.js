import * as dotenv from 'dotenv'
import { OpenAI } from 'openai'

import { TokenJS } from '../src'
dotenv.config()

const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
  {
    role: 'user',
    content: `Tell me a joke about the moon.`,
  },
]

const callLLM = async () => {
  const tokenjs = new TokenJS({
    baseURL: 'http://localhost:3000/api/',
  })
  const result = await tokenjs.chat.completions.create({
    // stream: true,
    provider: 'openai-compatible',
    model: 'gemini-1.5-pro',
    messages,
  })

  console.log(result.choices)

  // for await (const part of result) {
  // process.stdout.write(part.choices[0]?.delta?.content || "");
  // }
}

const callOpenAI = async () => {
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    dangerouslyAllowBrowser: true,
  })

  const result = await openai.chat.completions.create({
    messages,
    model: 'gpt-4o',
    stream: true,
  })

  for await (const part of result) {
    console.log(part.choices[0].finish_reason)
  }

  // console.log(result.choices[0].message.content)
}

// callOpenAI()
callLLM()
