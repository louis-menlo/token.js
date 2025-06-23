import OpenAI from 'openai'
import { Stream } from 'openai/streaming'

import {
  PerplexityModel,
  ProviderCompletionParams,
  RequestOptions,
} from '../chat/index.js'
import {
  CompletionResponse,
  StreamCompletionResponse,
} from '../userTypes/index.js'
import { BaseHandler } from './base.js'
import { InputError } from './types.js'

export const PERPLEXITY_PREFIX = 'perplexity/'

async function* streamPerplexity(
  response: Stream<OpenAI.Chat.Completions.ChatCompletionChunk>
): StreamCompletionResponse {
  for await (const chunk of response) {
    yield chunk
  }
}

export class PerplexityHandler extends BaseHandler<PerplexityModel> {
  async create(
    body: ProviderCompletionParams<'perplexity'>,
    options?: RequestOptions
  ): Promise<CompletionResponse | StreamCompletionResponse> {
    this.validateInputs(body)
    const apiKey = this.opts.apiKey ?? process.env.PERPLEXITY_API_KEY

    if (apiKey === undefined) {
      throw new InputError(
        'API key is required for Perplexity, define PERPLEXITY_API_KEY in your environment or specifty the apiKey option.'
      )
    }

    const openai = new OpenAI({
      ...this.opts,
      baseURL: 'https://api.perplexity.ai',
      apiKey,
      dangerouslyAllowBrowser: true,
    })

    const model = body.model.replace(PERPLEXITY_PREFIX, '')
    // Perplexity throws an error if the temperature equals two, so if the user sets it to 2, we
    // assign it to a marginally lower value.
    const temperature =
      body.temperature === 2 ? 2 - Number.EPSILON : body.temperature

    const request = {
      ...body,
      model,
      temperature,
    }

    if (request.stream) {
      const stream = await openai.chat.completions.create(request, options)
      return streamPerplexity(stream)
    } else {
      return openai.chat.completions.create(request, options)
    }
  }
}
